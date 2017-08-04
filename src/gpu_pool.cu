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

#define NFPGAS 48
#define NACCUMULATE 128

GpuPool::GpuPool(int poolid, InConfig config) : accumulate_(config.accumulate),
                                        avgfreq_(config.freqavg),
                                        avgtime_(config.timeavg),
                                        beamno_(0),
                                        codiflen_(config.codiflen),
                                        config_(config),
                                        dedispgulpsamples_(config.gulp),
                                        fftbatchsize_(config.nopols * config.nochans * config.accumulate * 128 / config.fftsize),
                                        fftedsize_(config.nopols * config.nochans * config.accumulate * 128 / config.fftsize * config.fftsize),
                                        fftpoints_(config.fftsize),
                                        filbits_(config.outbits),
                                        filchans_(config.filchans),
                                        gpuid_(config.gpuids[poolid]),
                                        gulpssent_(0),
                                        headlen_(config.headlen),
                                        ipstring_(config.ips[poolid]),
                                        // NOTE: There are config.nochans * config.accumulate * 128 8-byte words
                                        inbuffsize_(8  * config.nochans * config.accumulate * 128),
                                        inchans_(config.nochans),
                                        nopols_(config.nopols),
                                        noports_(config.noports),
                                        nostokes_(config.nostokes),
                                        nostreams_(config.nostreams),
                                        poolid_(poolid),
                                        ports_(config.ports),
                                        // NOTE: Quick hack to switch the scaling off
                                        scaled_(true),
                                        secondstorecord_(config.record),
                                        unpackedbuffersize_(config.nopols * config.nochans * config.accumulate * 128),
                                        verbose_(config.verbose) {

    // TODO: This statement doesn't make sense - we eiather have enough cores or not
    // NOTE: usethreads_ is not used anywhere
    usethreads_ = min(nostreams_ + 2, thread::hardware_concurrency());

    if (verbose_)
        PrintSafe("Starting GPU pool", gpuid_);
}

void GpuPool::Initialise(void) {

    struct bitmask *mask = numa_parse_nodestring((std::to_string(poolid_)).c_str());
    numa_bind(mask);

    signal(SIGINT, GpuPool::HandleSignal);
    cudaCheckError(cudaSetDevice(poolid_));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 10, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for the GPU pool", poolid_);
        exit(EXIT_FAILURE);     // affinity is critical for us
    }

    // NOTE: The output number of channels has to be divisible by 4
    // This is a requirement for the dedisp/Heimdall GPU memory access
    // In this case, any power of 2, greater than 4 works
    // TODO: Test whether there can be a better way of doing this
    // Using the closest lower power of 2 can lose us a lot of channels
    // filchans_ = 1 << (int)log2f(filchans_);

    if (verbose_)
        PrintSafe("GPU pool for device", gpuid_, "running on CPU", sched_getcpu());

    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    if (verbose_)
        PrintSafe("Preparing the memory on pool", poolid_, "...");

    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, config_.tsamp, config_.ftop, config_.foff, gpuid_));
    filbuffer_ = unique_ptr<FilterbankBuffer>(new FilterbankBuffer(gpuid_));

    framenumbers_ = new int[accumulate_ * nostreams_];
    std::fill(framenumbers_, framenumbers_ + accumulate_ * nostreams_, -1);
    gpustreams_ = new cudaStream_t[nostreams_];
    fftplans_ = new cufftHandle[nostreams_];

    // TODO: Can really remove this array
    // Wait until we put the final optimised verions of kernels
    int nokernels = 4;
    cudathreads_ = new unsigned int[nokernels];
    cudablocks_ = new unsigned int[nokernels];

    cudathreads_[0] = 7;
    cudathreads_[1] = fftpoints_ * avgtime_ * inchans_ / 42;
    cudathreads_[2] = inchans_;
    cudathreads_[3] = filchans_;

    cudablocks_[0] = 48;
    cudablocks_[1] = 42;
    cudablocks_[2] = 1;
    cudablocks_[3] = 1;

    // NOTE: Each stream will have its own incoming buffer to read from
    // NOTE: inchans_ / 7 as each packet receives 7 channels
    packperbuffer_ = NFPGAS * accumulate_ * nostreams_;
    hinbuffer_ = new unsigned char[inbuffsize_ * nostreams_];
    readybuffidx_ = new bool[NFPGAS * accumulate_ * nostreams_];
    std::fill(readybuffidx_, readybuffidx_ + NFPGAS * accumulate_ * nostreams_, 0);

    cudaCheckError(cudaHostAlloc((void**)&hstreambuffer_, inbuffsize_ * nostreams_ * sizeof(unsigned char), cudaHostAllocDefault));
    cudaCheckError(cudaMalloc((void**)&dstreambuffer_, inbuffsize_ * nostreams_ * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc((void**)&dunpackedbuffer_, unpackedbuffersize_ * nostreams_ * sizeof(cufftComplex)));
    cudaCheckError(cudaMalloc((void**)&dfftedbuffer_, fftedsize_ * nostreams_ * sizeof(cufftComplex)));

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


    /**
     * NOTE [Ewan]: We have hardcoded the extra portion of the buffer
     * to zero during debugging for Effelsberg. There is no reason to
     * believe that this can't be uncommented now, but I am leaving it
     * out as we are confident that the system is working right now.
     */
    //dedispextrasamples_ = dedispplan_->get_max_delay();
    dedispextrasamples_ = 0;
    dedispdispersedsamples_ = (size_t)dedispgulpsamples_ + dedispextrasamples_;
    //dedispnobuffers_ = (dedispdispersedsamples_ - 1) / dedispgulpsamples_ + 1;

    /**
     * Note [Ewan]: Same sentiment as above. This is commented out for debugging, but
     * can likely be renabled safely.
     */
    dedispnobuffers_  = 2;
    dedispbuffersize_ = dedispnobuffers_ * dedispgulpsamples_ + dedispextrasamples_;
    filbuffer_->Allocate(accumulate_, dedispnobuffers_, dedispextrasamples_, dedispgulpsamples_, dedispbuffersize_, filchans_, nostokes_, filbits_);
    dedispplan_->set_killmask(&config_.killmask[0]);

    // STAGE: PREPARE THE SINGLE PULSE SEARCH
    if (verbose_)
        PrintSafe("Setting up dedispersion and single pulse search on pool", poolid_, "...");

    SetSearchParams(singleparams_, config_);
    // NOTE: Commented out for the filterbank dump mode
    //hd_create_pipeline(&pipeline, params);
    // NOTE: Everything should be ready for single pulse search after this point

    // STAGE: start processing
    // FFT threads

    // NOTE: It has to be an array and I can't do anything about that
    fftsizes_[0] = (int)fftpoints_;

    for (int igstream = 0; igstream < nostreams_; igstream++) {
            cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
            cufftCheckError(cufftPlanMany(&fftplans_[igstream], 1, fftsizes_, NULL, 1, fftpoints_, NULL, 1, fftpoints_, CUFFT_C2C, fftbatchsize_));
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
    //hints.ai_protocol = IPPROTO_UDP

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

    // need deallocation in the dedisp buffer destructor as well
    filbuffer_->Deallocate();
    // this stuff is deleted in order it appears in the code
    delete [] framenumbers_;
    delete [] gpustreams_;
    delete [] cudathreads_;
    delete [] cudablocks_;
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
    //Note: changed to 10 for Pacifix machines (this should not be hardcoded)
    CPU_SET((int)(poolid_) * 10 + 1 + (int)(stream / 1), &cpuset);
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
    unsigned int skip = stream * unpackedbuffersize_;

    unsigned char **pfil = filbuffer_ -> GetFilPointer();

    ObsTime frametime;

    int skipread = stream * NFPGAS * NACCUMULATE;
    int skiptoend = (stream + 1) * NFPGAS * NACCUMULATE - 1;
    int nextstart;

    /**
     * Note [Ewan]: This was an important fix. Previously we had
     * nextstart = 128 in the else clause, which meant you only
     * needed one packet out of order to break the system
     */

    if (stream != 3) {
        nextstart = skiptoend + 128*NFPGAS;
    } else {
        nextstart = 128*NFPGAS-1;
    }
    bool endready = false;
    bool innext = false;

    while (working_) {
        endready = false;
        innext = false;
        for (int iidx = 0; iidx < 64; iidx++) {
            endready = endready || readybuffidx_[skiptoend - iidx];
            innext = innext || readybuffidx_[nextstart - iidx];
        }
        if (endready && innext) {
            for (int iidx = 0; iidx < 64; iidx++) {
                readybuffidx_[skiptoend - iidx] = false;
                readybuffidx_[nextstart - iidx] = false;
            }
            std::copy(hinbuffer_ + stream * inbuffsize_,  hinbuffer_ + stream * inbuffsize_ + inbuffsize_, hstreambuffer_ + stream * inbuffsize_);;
            for (int frameidx = 0; frameidx < accumulate_; frameidx++) {
                if (framenumbers_[stream * accumulate_ + frameidx] != -1) {
    		  frametime.framefromstart = framenumbers_[stream * accumulate_ + frameidx] - frameidx;
                  break;
                }
            }
            std::fill(framenumbers_ + stream * accumulate_, framenumbers_ + (stream + 1) * accumulate_, -1);
            frametime.startepoch = starttime_.startepoch;
            frametime.startsecond = starttime_.startsecond;
            cudaCheckError(cudaMemcpyAsync(dstreambuffer_ + stream * inbuffsize_, hstreambuffer_ + stream * inbuffsize_, inbuffsize_, cudaMemcpyHostToDevice, gpustreams_[stream]));
            UnpackKernel<<<48, 128, 0, gpustreams_[stream]>>>(reinterpret_cast<int2*>(dstreambuffer_ + stream * inbuffsize_), dunpackedbuffer_ + skip);
            cufftCheckError(cufftExecC2C(fftplans_[stream], dunpackedbuffer_ + skip, dfftedbuffer_ + skip, CUFFT_FORWARD));
	    // Note [Ewan]: bugfix, this kernel was being passed the unpacked rather than fft'd data.
            DetectScrunchKernel<<<2 * NACCUMULATE, 1024, 0, gpustreams_[stream]>>>(dfftedbuffer_ + skip, reinterpret_cast<float*>(pfil[0]), filchans_, dedispnobuffers_, dedispgulpsamples_, dedispextrasamples_, frametime.framefromstart);
            cudaStreamSynchronize(gpustreams_[stream]);
            cudaCheckError(cudaGetLastError());
            filbuffer_ -> UpdateFilledTimes(frametime);
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuPool::SendForDedispersion(void) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    //Note [Ewan]: multiply by 10 to match pacifix numa layout (should not be hardcoded)
    CPU_SET((int)(poolid_) * 10, &cpuset);
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
                headerfil.ra = config_.ra;
                headerfil.dec = config_.dec;
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
                //headerfil.tstart = GetMjd(sendtime.startepoch, sendtime.startsecond + 27 + sendtime.framefromstart * config_.tsamp);
                // TODO: This line doesn't work - fix this! Possible bug related to multiple time samples per frame
                headerfil.za = 0.0;
                headerfil.data_type = 1;
                headerfil.ibeam = beamno_;
                headerfil.machine_id = 8;
                headerfil.nbeams = 1;
                headerfil.nbits = filbits_;
                headerfil.nchans = filchans_;
                headerfil.nifs = 1;
                headerfil.telescope_id = 8;

                if (verbose_)
                    PrintSafe(ready - 1, "buffer ready on pool", poolid_);

                filbuffer_ -> SendToRam(ready, dedispstream_, (gulpssent_ % 2));
                filbuffer_ -> SendToDisk((gulpssent_ % 2), headerfil, config_.outdir);
                // TODO: Possible race condition
                gulpssent_++;

                if (verbose_)
                    PrintSafe("Filterbank", gulpssent_, "with MJD", headerfil.tstart, "for beam", beamno_, "on pool", poolid_, "saved");

                if ((int)(gulpssent_ * dedispdispersedsamples_ * config_.tsamp) >= secondstorecord_)
                    working_ = false;

            }   else {
                // perform the scaling
                // NOTE: Scaling breaks down when there is no data - division by a standard deviation of 0
                // TODO: Need to come up with a more clever way of dealing with that
                // filbuffer_->GetScaling(ready, dedispstream_, dmeans_, drstdevs_);
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
    //Note [Ewan]: multiply by 10 to match pacifix numa layout (should not be hardcoded)
    CPU_SET((int)(poolid_) * 10 + 1 + nostreams_ + (int)(portid / 3), &cpuset);
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
        // NOTE: Which stream buffer the data is saved to
        bufidx = (int)(frame / accumulate_) % nostreams_;
        // NOTE: Number of packets to skip to get to the start of the stream buffer
        bufidx *= NFPGAS * NACCUMULATE;
        // NOTE: Correct FPGA within the stream buffer
        bufidx += fpga * NACCUMULATE;
        // NOTE: Correct frame packet within the stream buffer
        bufidx += (frame % accumulate_);

        // bufidx = ((int)(frame / accumulate_) % nostreams_) * pack_per_worker_buf;
        // bufidx += (frame % accumulate_) * 48;

        framenumbers_[frame % (accumulate_ * nostreams_)] = frame;
        // bufidx += fpga;
        std::copy(receivebuffers_[portid] + headlen_, receivebuffers_[portid] + codiflen_ + headlen_, hinbuffer_ + codiflen_ * bufidx);



        readybuffidx_[bufidx] = true;
    }
}
