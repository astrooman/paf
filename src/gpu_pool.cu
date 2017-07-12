#include <algorithm>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <cufft.h>
#include <cuda.h>
#include <numa.h>
#include <pthread.h>

#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "filterbank_buffer.cuh"
#include "get_mjd.hpp"
#include "gpu_pool.cuh"
#include "heimdall/pipeline.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"
#include "pdif.hpp"
#include "print_safe.hpp"

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

using std::endl;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

bool GpuPool::working_ = true;

GpuPool::GpuPool(int poolid, InConfig config) : accumulate_(config.accumulate),
                                        avgfreq_(config.freqavg),
                                        avgtime_(config.timeavg),
                                        beamno_(0),
                                        codiflen_(config.codiflen),
                                        dedispgulpsamples_(config.gulp),
                                        fftbatchsize_(config.batch),
                                        fftedsize_(config.batch * config.fftsize * config.timeavg * config.nopols * config.accumulate),
                                        fftpoints_(config.fftsize),
                                        filchans_(config.filchans),
                                        freqscrunchedsize_((config.fftsize - 5) * config.batch  * config.accumulate / config.freqavg),
                                        gpuid_(config.gpuids[poolid]),
                                        gulpssent_(0),
                                        headlen_(config.headlen),
                                        ipstring_(config.ips[poolid]),
                                        inbuffsize_(8 * config.batch * config.fftsize * config.timeavg * config.accumulate),
                                        inchans_(config.nochans),
                                        nopols_(config.nopols),
                                        noports_(config.noports),
                                        nostokes_(config.nostokes),
                                        nostreams_(config.nostreams),
                                        poolid_(poolid),
                                        ports_(config.ports),
                                        // NOTE: 32 * 4 gives the starting 128 time samples, but this is not the correct implementation
                                        // TODO: Need to include the time averaging in the correct way
                                        rearrangedsize_(config.batch * config.fftsize * config.timeavg * config.nopols * config.accumulate),
                                        scaled_(false),
                                        secondstorecord_(config.record),
                                        timescrunchedsize_((config.fftsize - 5) * config.batch * config.accumulate),
                                        verbose_(config.verbose) {

    usethreads_ = min(nostreams_ + 2, thread::hardware_concurrency());

    config_ = config;

    if (verbose_)
        PrintSafe("Starting GPU pool", gpuid_);
}

void GpuPool::Initialise(void) {
    struct bitmask *mask = numa_parse_nodestring((std::to_string(poolid_)).c_str());
    numa_bind(mask);

    signal(SIGINT, GpuPool::HandleSignal);
    cudaCheckError(cudaSetDevice(poolid_));

    // NOTE: The output number of channels has to be divisible by 4
    // This is a requirement for the dedisp/Heimdall GPU memory access
    // In this case, any power of 2, greater than 4 works
    // TODO: Test whether there can be a better way of doing this
    // Using the closest lower power of 2 can lose us a lot of channels
    filchans_ = 1 << (int)log2f(filchans_);
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for the GPU pool", poolid_);
        exit(EXIT_FAILURE);     // affinity is critical for us
    }

    if (verbose_)
        PrintSafe("GPU pool for device", gpuid_, "running on CPU", sched_getcpu());

    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, config_.tsamp, config_.ftop, config_.foff, gpuid_));
    filbuffer_ = unique_ptr<FilterbankBuffer<float>>(new FilterbankBuffer<float>(gpuid_));

    framenumbers_ = new unsigned int[accumulate_ * nostreams_];
    gpustreams_ = new cudaStream_t[nostreams_];
    fftplans_ = new cufftHandle[nostreams_];

    int nokernels = 4;
    cudathreads_ = new unsigned int[nokernels];
    cudablocks_ = new unsigned int[nokernels];

    // TODO: Put optimised versions of these kernels in
    cudathreads_[0] = 7;
    cudathreads_[1] = fftpoints_ * avgtime_ * fftbatchsize_ / 42;
    cudathreads_[2] = inchans_;
    cudathreads_[3] = filchans_;
    //cudathreads_[3] = filchans_;

    cudablocks_[0] = 48;
    cudablocks_[1] = 42;
    cudablocks_[2] = 1;
    cudablocks_[3] = 1;

    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    if (verbose_)
        PrintSafe("Preparing the memory on pool", poolid_, "...");

    arrangechandesc_ = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());

    arrange2darray_ = new cudaArray*[nostreams_];
    arrangetexobj_ = new cudaTextureObject_t[nostreams_];
    arrangeresdesc_ = new cudaResourceDesc[nostreams_];
    arrangetexdesc_ = new cudaTextureDesc[nostreams_];
    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaMallocArray(&(arrange2darray_[igstream]), &arrangechandesc_, 7, (fftbatchsize_  / 7) * fftpoints_ * avgtime_ * accumulate_));

        memset(&(arrangeresdesc_[igstream]), 0, sizeof(cudaResourceDesc));
        arrangeresdesc_[igstream].resType = cudaResourceTypeArray;
        arrangeresdesc_[igstream].res.array.array = arrange2darray_[igstream];

        memset(&(arrangetexdesc_[igstream]), 0, sizeof(cudaTextureDesc));
        arrangetexdesc_[igstream].addressMode[0] = cudaAddressModeClamp;
        arrangetexdesc_[igstream].filterMode = cudaFilterModePoint;
        arrangetexdesc_[igstream].readMode = cudaReadModeElementType;

        arrangetexobj_[igstream] = 0;
        cudaCheckError(cudaCreateTextureObject(&(arrangetexobj_[igstream]), &(arrangeresdesc_[igstream]), &(arrangetexdesc_[igstream]), NULL));
    }

    // NOTE: It has to be an array and I can't do anything about that
    fftsizes_[0] = (int)fftpoints_;

    // NOTE: Each stream will have its own incoming buffer to read from
    packperbuffer_ = fftbatchsize_ / 7 * accumulate_ * nostreams_;
    hinbuffer_ = new unsigned char[inbuffsize_ * nostreams_];
    readybuffidx_ = new bool[packperbuffer_]();
    cudaCheckError(cudaHostAlloc((void**)&hstreambuffer_, inbuffsize_ * nostreams_ * sizeof(unsigned char), cudaHostAllocDefault));
    cudaCheckError(cudaMalloc((void**)&dstreambuffer_, rearrangedsize_ * nostreams_ * sizeof(cufftComplex)));
    cudaCheckError(cudaMalloc((void**)&dfftedbuffer_, fftedsize_ * nostreams_ * sizeof(cufftComplex)));

    htimescrunchedbuffer_ = new float*[nostreams_];
    hfreqscrunchedbuffer_ = new float*[nostreams_];

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaMalloc((void**)&htimescrunchedbuffer_[igstream], timescrunchedsize_ * nostokes_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hfreqscrunchedbuffer_[igstream], freqscrunchedsize_ * nostokes_ * sizeof(float)));
    }

    cudaCheckError(cudaMalloc((void**)&dtimescrunchedbuffer_, nostreams_ * sizeof(float*)));
    cudaCheckError(cudaMalloc((void**)&dfreqscrunchedbuffer_, nostreams_ * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dtimescrunchedbuffer_, htimescrunchedbuffer_, nostreams_ * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dfreqscrunchedbuffer_, hfreqscrunchedbuffer_, nostreams_ * sizeof(float*), cudaMemcpyHostToDevice));

    hmeans_ = new float*[nostokes_];
    hrstdevs_ = new float*[nostokes_];

    // NOTE: Need to be careful what to fill the starting values with
    // We want to have the original data after the scaling in the fist run
    // So we can actually obtain the first scaling factors
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hmeans_[istoke], filchans_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hrstdevs_[istoke], filchans_ * sizeof(float)));
    }

    cudaCheckError(cudaMalloc((void**)&dmeans_, nostokes_ * sizeof(float*)));
    cudaCheckError(cudaMalloc((void**)&drstdevs_, nostokes_ * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dmeans_, hmeans_, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(drstdevs_, hrstdevs_, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice));

    InitScaleFactors<<<1,filchans_,0,0>>>(dmeans_, drstdevs_, nostokes_);
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaGetLastError());

    // STAGE: PREPARE THE DEDISPERSION
    // NOTE: generate_dm_list(dm_start, dm_end, width, tol)
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials
    dedispplan_->generate_dm_list(config_.dmstart, config_.dmend, 64.0f, 1.10f);
    dedispextrasamples_ = dedispplan_->get_max_delay();
    dedispdispersedsamples_ = (size_t)dedispgulpsamples_ + dedispextrasamples_;
    dedispnobuffers_ = (dedispdispersedsamples_ - 1) / dedispgulpsamples_ + 1;
    dedispbuffersize_ = dedispnobuffers_ * dedispgulpsamples_ + dedispextrasamples_;
    filbuffer_->Allocate(accumulate_, dedispnobuffers_, dedispextrasamples_, dedispgulpsamples_, dedispbuffersize_, filchans_, nostokes_);
    dedispplan_->set_killmask(&config_.killmask[0]);

    // STAGE: PREPARE THE SINGLE PULSE SEARCH
    if (verbose_)
        PrintSafe("Setting up dedispersion and single pulse search on pool", poolid_, "...");

    SetSearchParams(singleparams_, config_);
    // NOTE: Commented out for the filterbank dump mode
    //hd_create_pipeline(&pipeline, params);
    // NOTE: everything should be ready for single pulse search after this point

    // STAGE: start processing
    // FFT threads
    for (int igstream = 0; igstream < nostreams_; igstream++) {
            cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
            cufftCheckError(cufftPlanMany(&fftplans_[igstream], 1, fftsizes_, NULL, 1, fftpoints_, NULL, 1, fftpoints_, CUFFT_C2C, fftbatchsize_ * avgtime_ * nopols_ * accumulate_));
            cufftCheckError(cufftSetStream(fftplans_[igstream], gpustreams_[igstream]));
            gputhreads_.push_back(thread(&GpuPool::FilterbankData, this, igstream));
    }

    cudaCheckError(cudaStreamCreate(&dedispstream_));
    gputhreads_.push_back(thread(&GpuPool::SendForDedispersion, this));

    // STAGE: Networking
    if (verbose_)
        PrintSafe("Setting up networking on pool", poolid_, "...");

    int netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    filedesc_ = new int[noports_];

    receivebuffers_ = new unsigned char*[noports_];
    for (int iport = 0; iport < noports_; iport++)
        receivebuffers_[iport] = new unsigned char[codiflen_ + headlen_];

    std::ostringstream oss;
    std::string strport;

    // all the magic happens here
    for (int iport = 0; iport < noports_; iport++) {
        // TODO: Read port numbers from the config file
        oss.str("");
        oss << ports_.at(iport);
        strport = oss.str();

        if ((netrv = getaddrinfo(ipstring_.c_str(), strport.c_str(), &hints, &servinfo)) != 0)
            PrintSafe("getaddrinfo() error:", gai_strerror(netrv), "on pool", poolid_);

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((filedesc_[iport] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                PrintSafe("Socket error on pool", poolid_);
                continue;
            }

            if (bind(filedesc_[iport], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(filedesc_[iport]);
                PrintSafe("Bind error on pool", poolid_);
                continue;
            }
            break;
        }

        if (tryme == NULL)
            PrintSafe("Failed to bind to the socket", ports_.at(iport), "on pool", poolid_);
    }

    for (int iport = 0; iport < noports_; iport++)
        receivethreads_.push_back(thread(&GpuPool::ReceiveData, this, iport, ports_.at(iport)));

}

GpuPool::~GpuPool(void) {

    for(int ithread = 0; ithread < gputhreads_.size(); ithread++)
        gputhreads_[ithread].join();

    for (int ithread = 0; ithread < noports_; ithread++)
        receivethreads_[ithread].join();

    //save the scaling factors before quitting
    if (scaled_) {
        string scalename = config_.outdir + "/scale_beam_" + std::to_string(beamno_) + ".dat";
        std::fstream scalefile(scalename.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (scalefile) {
            float *means = new float[filchans_];
            float *stdevs = new float[filchans_];
            for (int istoke = 0; istoke < nostokes_; istoke++) {
                cudaCheckError(cudaMemcpy(means, hmeans_[istoke], filchans_ * sizeof(float), cudaMemcpyDeviceToHost));
                cudaCheckError(cudaMemcpy(stdevs, hrstdevs_[istoke], filchans_ * sizeof(float), cudaMemcpyDeviceToHost));
                for (int jj = 0; jj < filchans_; jj++) {
                    scalefile << means[jj] << " " << stdevs[jj] << endl;
                }
                scalefile << endl << endl;
            }
        }
        scalefile.close();
    }
    // cleaning up the stuff
    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaDestroyTextureObject(arrangetexobj_[igstream]));
        cudaCheckError(cudaFreeArray(arrange2darray_[igstream]));
    }

    // need deallocation in the dedisp buffer destructor as well
    filbuffer_->Deallocate();
    // this stuff is deleted in order it appears in the code
    delete [] framenumbers_;
    delete [] gpustreams_;
    delete [] cudathreads_;
    delete [] cudablocks_;
    delete [] arrange2darray_;
    delete [] arrangetexobj_;
    delete [] arrangeresdesc_;
    delete [] arrangetexdesc_;
    delete [] hinbuffer_;
    delete [] readybuffidx_;
    delete [] filedesc_;
    for (int iport = 0; iport < noports_; iport++) {
        delete [] receivebuffers_[iport];
    }
    delete [] receivebuffers_;


    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFree(hmeans_[istoke]));
        cudaCheckError(cudaFree(hrstdevs_[istoke]));
    }

    delete [] hmeans_;
    delete [] hrstdevs_;
    cudaCheckError(cudaFree(dmeans_));
    cudaCheckError(cudaFree(drstdevs_));

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        // TODO: cudaFree() the scrunched data buffers
        cudaCheckError(cudaFree(hfreqscrunchedbuffer_[igstream]));
        cudaCheckError(cudaFree(htimescrunchedbuffer_[igstream]));
    }

    delete [] htimescrunchedbuffer_;
    delete [] hfreqscrunchedbuffer_;
    cudaCheckError(cudaFree(dfreqscrunchedbuffer_));
    cudaCheckError(cudaFree(dtimescrunchedbuffer_));

    cudaCheckError(cudaFree(dstreambuffer_));
    cudaCheckError(cudaFree(dfftedbuffer_));
    cudaCheckError(cudaFreeHost(hstreambuffer_));

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cufftCheckError(cufftDestroy(fftplans_[igstream]));
    }

    delete [] fftplans_;
}

void GpuPool::HandleSignal(int signum) {

    PrintSafe("Captured the signal!\nWill now terminate!\n");
    working_ = false;

}

void GpuPool::FilterbankData(int stream) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + (int)(stream / 1), &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for stream", stream, "on pool", poolid_);
        exit(EXIT_FAILURE);
    }

    if (verbose_)
        PrintSafe("Starting worker", stream, "on pool", poolid_, "on CPU", sched_getcpu());

    cudaSetDevice(gpuid_);
    dim3 rearrange_b(1,48,1);
    dim3 rearrange_t(7,1,1);
    unsigned int skip = stream * rearrangedsize_;

    float **pfil = filbuffer_ -> GetFilPointer();
    float **pdfil;
    cudaMalloc((void**)&pdfil, nostokes_ * sizeof(float*));
    cudaMemcpy(pdfil, pfil, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice);

    ObsTime frametime;

    float *ptimescrunchedbuffer_ = htimescrunchedbuffer_[stream];
    float *pfreqscrunchedbuffer_ = hfreqscrunchedbuffer_[stream];

    // TODO: This has to be simplified
    int skipread = stream * (packperbuffer_ / nostreams_);
    int skiptoend = (stream + 1) * (packperbuffer_ / nostreams_) - 1;
    int nextstart;
    if (stream != 3) {
        nextstart = skiptoend + 24;
    } else {
        nextstart = 23;
    }
    bool endready = false;
    bool innext = false;
    while (working_) {
        endready = false;
        innext = false;
        for (int iidx = 0; iidx < 4; iidx++) {
            endready = endready || readybuffidx_[skiptoend - iidx];
            innext = innext || readybuffidx_[nextstart - iidx];
        }
        if (endready && innext) {
            for (int iidx = 0; iidx < 4; iidx++) {
                readybuffidx_[skiptoend - iidx] = false;
                readybuffidx_[nextstart - iidx] = false;
            }
            std::copy(hinbuffer_ + stream * inbuffsize_,  hinbuffer_ + stream * inbuffsize_ + inbuffsize_, hstreambuffer_ + stream * inbuffsize_);;
            for (int frameidx = 0; frameidx < accumulate_; frameidx++) {
                if (framenumbers_[stream * accumulate_ + frameidx] != 0) {
                    frametime.framefromstart = framenumbers_[stream * accumulate_ + frameidx];
                    break;
                }
            }
            for (int frameidx = 0; frameidx < accumulate_; frameidx++)
                framenumbers_[stream * accumulate_ + frameidx] = 0;

            frametime.startepoch = starttime_.startepoch;
            frametime.startsecond = starttime_.startsecond;

            cudaCheckError(cudaMemcpyToArrayAsync(arrange2darray_[stream], 0, 0, hstreambuffer_ + stream * inbuffsize_, inbuffsize_, cudaMemcpyHostToDevice, gpustreams_[stream]));
            RearrangeKernel<<<rearrange_b, rearrange_t, 0, gpustreams_[stream]>>>(arrangetexobj_[stream], dstreambuffer_ + skip, accumulate_);
            cufftCheckError(cufftExecC2C(fftplans_[stream], dstreambuffer_ + skip, dfftedbuffer_ + skip, CUFFT_FORWARD));
            GetPowerAddTimeKernel<<<48, 27, 0, gpustreams_[stream]>>>(dfftedbuffer_ + skip, ptimescrunchedbuffer_, timescrunchedsize_, avgtime_, accumulate_);
            AddChannelsScaleKernel<<<cudablocks_[3], cudathreads_[3], 0, gpustreams_[stream]>>>(ptimescrunchedbuffer_, pdfil, filchans_, dedispgulpsamples_, dedispbuffersize_, dedispnobuffers_, timescrunchedsize_, avgfreq_, frametime.framefromstart, accumulate_, dmeans_, drstdevs_);
            cudaStreamSynchronize(gpustreams_[stream]);
            cudaCheckError(cudaGetLastError());
            filbuffer_ -> UpdateFilledTimes(frametime);
            //working_ = false;
        } else {
            std::this_thread::yield();
        }
    }

    cudaCheckError(cudaFree(pdfil));
}

void GpuPool::SendForDedispersion(void) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8, &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[nostreams_].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for dedisp thread on pool", poolid_);
        exit(EXIT_FAILURE);
    }

    ObsTime sendtime;

    cudaCheckError(cudaSetDevice(gpuid_));
    if (verbose_)
        PrintSafe("Dedisp thread up and running on pool", poolid_, "...");

    int ready{0};
    while(working_) {
        ready = filbuffer_->CheckIfReady();
        if (ready) {
            if (scaled_) {
                // TODO: Prefil the header with non-changing infomation
                header_f headerfil;
                headerfil.raw_file = "tastytastytest";
                headerfil.source_name = "J1641-45";
                headerfil.az = 0.0;
                headerfil.ra = config.ra;
                headerfil.dec = config.dec;
                // for channels in decreasing order
                headerfil.fch1 = 1173.0 - (16.0 / 27.0) + filchans_ * config_.foff;
                headerfil.foff = -1.0 * abs(config_.foff);
                // for channels in increasing order
                // headerfil.fch1 = 1173.0 - (16.0 / 27.0);
                // headerfil.foff = config_.foff;
                headerfil.rdm = 0.0;
                headerfil.tsamp = config_.tsamp;
                // TODO: this totally doesn't work when something is skipped
                headerfil.tstart = GetMjd(starttime_.startepoch, starttime_.startsecond + 27 + (gulpssent_ + 1)* dedispgulpsamples_ * config_.tsamp);
                sendtime = filbuffer_->GetTime(ready-1);
                headerfil.tstart = GetMjd(sendtime.startepoch, sendtime.startsecond + 27 + sendtime.framefromstart * config_.tsamp);
                headerfil.za = 0.0;
                headerfil.data_type = 1;
                headerfil.ibeam = beamno_;
                headerfil.machine_id = 2;
                headerfil.nbeams = 1;
                headerfil.nbits = 8;
                headerfil.nchans = filchans_;
                headerfil.nifs = 1;
                headerfil.telescope_id = 2;

                if (verbose_)
                    PrintSafe(ready - 1, "buffer ready on pool", poolid_);

                filbuffer_ -> SendToRam(d_dedisp, ready, dedispstream_, (gulpssent_ % 2));
                filbuffer_ -> SendToDisk((gulpssent_ % 2), headerfil, config_.outdir);
                gulpssent_++;

                if (verbose_)
                    PrintSafe("Filterbank", gulpssent_, "with MJD", headerfil.tstart, "for beam", beamno_, "on pool", poolid_, "saved");

                if ((int)(gulpssent_ * dedispdispersedsamples_ * config_.tsamp) >= secondstorecord_)
                    working_ = false;

            }   else {
                // perform the scaling
                // NOTE: Scaling breaks down when there is no data - division by a standard deviation of 0
                // TODO: Need to come up with a more clever way of dealing with that
                filbuffer_->GetScaling(ready, dedispstream_, dmeans_, drstdevs_);
                cudaCheckError(cudaGetLastError());
                scaled_ = true;
                ready = 0;

                if (verbose_)
                    PrintSafe("Scaling factors have been obtained on pool", poolid_);
            }

        } else {
            std::this_thread::yield();
        }
    }
}

void GpuPool::ReceiveData(int portid, int recport) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + nostreams_ + (int)(portid / 3), &cpuset);
    int retaff = pthread_setaffinity_np(receivethreads_[portid].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for receive thread on port", recport, "on pool", poolid_);
        exit(EXIT_FAILURE);
    }

    if (verbose_)
        PrintSafe("Receive thread on port", recport, "on pool", poolid_, "running on CPU", sched_getcpu());

    sockaddr_storage senderaddr;
    memset(&senderaddr, 0, sizeof(senderaddr));
    socklen_t addrlen;
    memset(&addrlen, 0, sizeof(addrlen));

    const int pack_per_worker_buf = packperbuffer_ / nostreams_;
    int numbytes{0};
    short fpga{0};
    short bufidx{0};

    int frame{0};
    int refsecond{0};
    int group{0};

    if (portid == 0) {
        unsigned char *tmpbuffer = receivebuffers_[0];
        numbytes = recvfrom(filedesc_[portid], receivebuffers_[portid], codiflen_ + headlen_ - 1, 0, (struct sockaddr*)&senderaddr, &addrlen);
        starttime_.startepoch = (int)(tmpbuffer[12] >> 2);
        starttime_.startsecond = (int)(tmpbuffer[3] | (tmpbuffer[2] << 8) | (tmpbuffer[1] << 16) | ((tmpbuffer[0] & 0x3f) << 24));
        beamno_ = (int)(tmpbuffer[23] | (tmpbuffer[22] << 8));
    }

    while (true) {
        if ((numbytes = recvfrom(filedesc_[portid], receivebuffers_[portid], codiflen_ + headlen_ - 1, 0, (struct sockaddr*)&senderaddr, &addrlen)) == -1)
            PrintSafe("recvfrom error on port", recport, "on pool", poolid_, "with code", errno);

        if (numbytes == 0)
            continue;
        frame = (int)(receivebuffers_[portid][7] | (receivebuffers_[portid][6] << 8) | (receivebuffers_[portid][5] << 16) | (receivebuffers_[portid][4] << 24));
        // NOTE: Wait until the start of the 27s boundary
        if (frame == 0) {
            break;
        }
    }

    while(working_) {
        if ((numbytes = recvfrom(filedesc_[portid], receivebuffers_[portid], codiflen_ + headlen_ - 1, 0, (struct sockaddr*)&senderaddr, &addrlen)) == -1)
            PrintSafe("recvfrom error on port", recport, "on pool", poolid_, "with code", errno);

        if (numbytes == 0)
            continue;
        refsecond = (int)(receivebuffers_[portid][3] | (receivebuffers_[portid][2] << 8) | (receivebuffers_[portid][1] << 16) | ((receivebuffers_[portid][0] & 0x3f) << 24));
        frame = (int)(receivebuffers_[portid][7] | (receivebuffers_[portid][6] << 8) | (receivebuffers_[portid][5] << 16) | (receivebuffers_[portid][4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&senderaddr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&senderaddr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frame = frame + (refsecond - starttime_.startsecond - 27) / 27 * 250000;

        // TODO: Add some mininal missing frames checks later anyway
        bufidx = ((int)(frame / accumulate_) % nostreams_) * pack_per_worker_buf;
        bufidx += (frame % accumulate_) * 48;
        framenumbers_[frame % (accumulate_ * nostreams_)] = frame;
        bufidx += fpga;
        std::copy(receivebuffers_[portid] + headlen_, receivebuffers_[portid] + codiflen_ + headlen_, hinbuffer_ + codiflen_ * bufidx);
        readybuffidx_[bufidx] = true;
    }
}
