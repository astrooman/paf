#include <algorithm>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <cufft.h>
#include <cuda.h>
#include <numa.h>
#include <pthread.h>
#include <thrust/device_vector.h>

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

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

using std::cout;
using std::endl;
using std::lock_guard;
using std::mutex;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

#define BYTES_PER_WORD 8
#define HEADER 64
#define WORDS_PER_PACKET 896
#define BUFLEN 7232

bool GpuPool::working_ = true;

void SafeCout(string instring) {
    mutex printmutex;
    lock_guard<mutex> guard(printmutex);
    cout << instring << endl;
}

// TODO: Could I do it this way?
std::ostream& operator<<(std::ostream &ostr, string outstr) {

}

GpuPool::GpuPool(int poolid, InConfig config) : accumulate_(config.accumulate),
                                        beamno_{0},
                                        dedispgulpsamples_(config.gulp),
                                        filchans_(config.filchans),
                                        gpuid_(config.gpuids[poolid]),
                                        ipstring_(config.ips[poolid]),
                                        fftedsize_(config.batch * config.fftsize * config.timesavg * config.npols * config.accumulate),
                                        freqscrunchedsize_((config.fftsize - 5) * config.batch  * config.accumulate / config.freqavg),
                                        highest_buf(0),
                                        inbuffsize_(8 * config.batch * config.fftsize * config.timesavg * config.accumulate),
                                        inchans_(config.nchans),
                                        fftbatchsize_(config.batch),
                                        fftpoints_(config.fftsize),
                                        avgtime_(config.timesavg),
                                        avgfreq_(config.freqavg),
                                        nopols_(config.npols),
                                        noports_(config.noports),
                                        nostokes_(config.stokes),
                                        nostreams_(config.streamno),
                                        poolid_(poolid),
                                        ports_(config.ports),
                                        // NOTE: 32 * 4 gives the starting 128 time samples, but this is not the correct implementation
                                        // TODO: Need to include the time averaging in the correct way
                                        rearrangedsize_(config.batch * config.fftsize * config.timesavg * config.npols * config.accumulate),
                                        timescrunchedsize_((config.fftsize - 5) * config.batch * config.accumulate),
                                        gulps_sent(0),
                                        gulps_processed(0),
                                        scaled_(false),
                                        verbose_(config.verbose),
                                        record_(config.record),
					packcount(0)

{
    usethreads_ = min(nostreams_ + 2, thread::hardware_concurrency());

    config_ = config;

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting GPU pool " << gpuid_ << endl;
	    cout.flush();
        cout_guard.unlock();
    }
}

void GpuPool::Initialise(void)
{
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

    // TODO: Use a function for priting things out with a mutex
    // Variadic templates?
    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for the GPU pool " << gpuid_ << endl;
        cout_guard.unlock();
    }

    if(verbose_) {
        cout_guard.lock();
        cout << "GPU pool for device " << gpuid_ << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, config_.tsamp, config_.ftop, config_.foff, gpuid_));
    filbuffer_ = unique_ptr<FilterbankBuffer<float>>(new FilterbankBuffer<float>(gpuid_));

    frame_times = new int[accumulate_ * nostreams_];
    gpustreams_ = new cudaStream_t[nostreams_];
    fftplans_ = new cufftHandle[nostreams_];

    int nokernels = 4;
    cudathreads_ = new unsigned int[nokernels];
    cudablocks_ = new unsigned int[nokernels];

    // TODO: The Rearrange() kernel has to be optimised
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
        cout << "Preparing the memory..." << endl;

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
    // TODO: Remove device vectors where possible
    dv_time_scrunch.resize(nostreams_);
    dv_freq_scrunch.resize(nostreams_);

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        dv_time_scrunch[igstream].resize(timescrunchedsize_ * nostokes_);
        dv_freq_scrunch[igstream].resize(freqscrunchedsize_ * nostokes_);
    }
    // scaling factors memory
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
    dedispplan_->generate_dm_list(config_.dstart, config_.dend, 64.0f, 1.10f);
    dedispextrasamples_ = dedispplan_->get_max_delay();
    dedispdispersedsamples_ = (size_t)dedispgulpsamples_ + dedispextrasamples_;
    dedispnobuffers_ = (dedispdispersedsamples_ - 1) / dedispgulpsamples_ + 1;
    dedispbuffersize_ = dedispnobuffers_ * dedispgulpsamples_ + dedispextrasamples_;
    if (verbose_)
        cout << "Total buffer size: " << dedispbuffersize_ << endl;
    filbuffer_->Allocate(accumulate_, dedispnobuffers_, dedispextrasamples_, dedispgulpsamples_, dedispbuffersize_, filchans_, nostokes_);
    dedispplan_->set_killmask(&config_.killmask[0]);

    // STAGE: PREPARE THE SINGLE PULSE SEARCH
    if (verbose_)
        cout << "Setting up dedispersion and single pulse search..." << endl;

    set_search_params(params, config_);
    //commented out for the filterbank dump mode
    //hd_create_pipeline(&pipeline, params);
    // everything should be ready for single pulse search after this point

    // STAGE: start processing
    // FFT threads
    for (int ii = 0; igstream < nostreams_; igstream++) {
            cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
            cufftCheckError(cufftPlanMany(&myplans[igstream], 1, fftsizes_, NULL, 1, fftpoints_, NULL, 1, fftpoints_, CUFFT_C2C, fftbatchsize_ * avgtime_ * nopols_ * _));
            cufftCheckError(cufftSetStream(myplans[igstream], gpustreams_[igstream]));
            gputhreads_.push_back(thread(&GpuPool::FilterbankData, this, igstream));
    }

    cudaCheckError(cudaStreamCreate(&dedispstream_));
    gputhreads_.push_back(thread(&GpuPool::SendForDedispersion, this));

    // STAGE: Networking
    if (verbose_)
        cout << "Setting up networking..." << endl;

    memset(&start_time, 0, sizeof(start_time)) ;
    int netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    filedesc_ = new int[noports_];

    receivebuffers_ = new unsigned char*[noports_];
    for (int iport = 0; iport < noports_; iport++)
        receivebuffers_[iport] = new unsigned char[BUFLEN];

    std::ostringstream oss;
    std::string strport;

    // all the magic happens here
    for (int iport = 0; iport < noports_; iport++) {
        // TODO: Read port numbers from the config file
        oss.str("");
        oss << 17100 + iport;
        strport = oss.str();

        if((netrv = getaddrinfo(ipstring_.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
            cout_guard.lock();
            cerr <<  "getaddrinfo() error: " << gai_strerror(netrv) << endl;
            cout_guard.unlock();
        }

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((filedesc_[iport] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                cout_guard.lock();
                cerr << "Socket error\n";
                cout_guard.unlock();
                continue;
            }

            if(bind(filedesc_[iport], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(filedesc_[iport]);
                cout_guard.lock();
                cerr << "Bind error\n";
                cout_guard.unlock();
                continue;
            }
            break;
        }

        if (tryme == NULL) {
            cout_guard.lock();
            cerr << "Failed to bind to the socket " << 17100 + iport << "\n";
            cout_guard.unlock();
        }
    }

    int bufres{4*1024*1024};    // 4MB

    for (int iport = 0; iport < noports_; iport++) {
        if(setsockopt(filedesc_[iport], SOL_SOCKET, SO_RCVBUF, (char *)&bufres, sizeof(bufres)) != 0) {
            cout_guard.lock();
            cerr << "Setsockopt error on port " << 17100 + iport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
    }

    for (int iport = 0; iport < noports_; iport++)
        receivethreads_.push_back(thread(&GpuPool::ReceiveData, this, ports_.at(iport)));

}

GpuPool::~GpuPool(void)
{
    // TODO: clear the memory properly
    if (verbose_)
        cout << "Calling destructor" << endl;

    for(int ithread = 0; ithread < gputhreads_.size(); ithread++)
        gputhreads_[ithread].join();

    for (int ithread = 0; ithread < noports_; ithread++)
        receive_threads[ithread].join();

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
    filbuffer_->deallocate();
    // this stuff is deleted in order it appears in the code
    delete [] frame_times;
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


    cudaCheckError(cudaFree(dstreambuffer_));
    cudaCheckError(cudaFree(dfftedbuffer_));
    cudaCheckError(cudaFreeHost(hstreambuffer_));

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cufftCheckError(cufftDestroy(myplans[igstream]));
    }

    delete [] myplans;
}

void GpuPool::HandleSignal(int signum) {

    cout << "Captured the signal\nWill now terminate!\n";
    working_ = false;
}

void GpuPool::FilterbankData(int stream)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + (int)(stream / 1), &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for stream " << stream << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting worker " << gpuid_ << ":" << stream << " on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    cudaSetDevice(gpuid_);
    dim3 rearrange_b(1,48,1);
    dim3 rearrange_t(7,1,1);
    unsigned int skip = stream * rearrangedsize_;

    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

    float **p_fil = filbuffer_->get_pfil();
    float **pd_fil;
    cudaMalloc((void**)&pd_fil, nostokes_ * sizeof(float *));
    cudaMemcpy(pd_fil, p_fil, nostokes_ * sizeof(float *), cudaMemcpyHostToDevice);

    ObsTime frametime;

    int skip_read = stream * (packperbuffer_ / nostreams_);
    int skip_to_end = (stream + 1) * (packperbuffer_ / nostreams_) - 1;
    int next_start;
    if (stream != 3) {
        next_start = skip_to_end + 24;
    } else {
        next_start = 23;
    }
    bool endready = false;
    bool innext = false;
    while (working_) {
        endready = false;
        innext = false;
        for (int iidx = 0; iidx < 4; iidx++) {
            //cout << readybuffidx_[skip_to_end - iidx] << " " << readybuffidx_[skip_to_end + 24 - iidx] << endl;
            endready = endready || readybuffidx_[skip_to_end - iidx];
            innext = innext || readybuffidx_[next_start - iidx];
        }
        if (endready && innext) {
            for (int iidx = 0; iidx < 4; iidx++) {
                readybuffidx_[skip_to_end - iidx] = false;
                readybuffidx_[next_start - iidx] = false;
            }
            std::copy(hinbuffer_ + stream * inbuffsize_,  hinbuffer_ + stream * inbuffsize_ + inbuffsize_, hstreambuffer_ + stream * inbuffsize_);;
            for (int frameidx = 0; frameidx < accumulate_; frameidx++) {
                if (frame_times[stream * accumulate_ + frameidx] != 0) {
                    frametime.framefromstart = frame_times[stream * accumulate_ + frameidx];
                    break;
                }
            }
            for (int frameidx = 0; frameidx < accumulate_; frameidx++)
                frame_times[stream * accumulate_ + frameidx] = 0;

            frametime.startepoch = starttime.startepoch;
            frametime.startsecond = starttime.startsecond;
            cudaCheckError(cudaMemcpyToArrayAsync(arrange2darray_[stream], 0, 0, hstreambuffer_ + stream * inbuffsize_, inbuffsize_, cudaMemcpyHostToDevice, gpustreams_[stream]));
            RearrangeKernel<<<rearrange_b, rearrange_t, 0, gpustreams_[stream]>>>(arrangetexobj_[stream], dstreambuffer_ + skip, accumulate_);
            cufftCheckError(cufftExecC2C(myplans[stream], dstreambuffer_ + skip, dfftedbuffer_ + skip, CUFFT_FORWARD));
            GetPowerAddTimeKernel<<<48, 27, 0, gpustreams_[stream]>>>(dfftedbuffer_ + skip, pdv_time_scrunch, timescrunchedsize_, avgtime_, accumulate_);
            AddChannelsScaleKernel<<<cudablocks_[3], cudathreads_[3], 0, gpustreams_[stream]>>>(pdv_time_scrunch, pd_fil, filchans_, dedispgulpsamples_, dedispbuffersize_, dedispnobuffers_, timescrunchedsize_, avgfreq_, frametime.framefromstart, accumulate_, dmeans_, drstdevs_);
            cudaStreamSynchronize(gpustreams_[stream]);
            cudaCheckError(cudaGetLastError());
            filbuffer_ -> UpdateFilledTimes(frame_time);
            //working_ = false;
        } else {
            std::this_thread::yield();
        }
    }

    cudaFree(pd_fil);
}

void GpuPool::SendForDedispersion(void)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8, &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[nostreams_].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cout_guard.lock();
        cout << "Error setting thread affinity for dedisp thread" << endl;
        cout_guard.unlock();
    }

    ObsTime sendtime;

    cudaCheckError(cudaSetDevice(gpuid_));
    if (verbose_)
        cout << "Dedisp thread up and running..." << endl;
    int ready{0};
    while(working_) {
        ready = filbuffer_->ready();
        if (ready) {
            if (scaled_) {
                // TODO: Prefil the header with non-changing infomation
                header_f headerfil;
                headerfil.raw_file = "tastytastytest";
                headerfil.source_name = "J1641-45";
                headerfil.az = 0.0;
                headerfil.dec = 0.0;
                // for channels in decreasing order
                headerfil.fch1 = 1173.0 - (16.0 / 27.0) + filchans_ * config_.foff;
                headerfil.foff = -1.0 * abs(config_.foff);
                // for channels in increasing order
                // headerfil.fch1 = 1173.0 - (16.0 / 27.0);
                // headerfil.foff = config_.foff;
                headerfil.ra = 0.0;
                headerfil.rdm = 0.0;
                headerfil.tsamp = config_.tsamp;
                // TODO: this totally doesn't work when something is skipped
                headerfil.tstart = get_mjd(start_time.start_epoch, start_time.start_second + 27 + (gulps_sent + 1)* dedispgulpsamples_ * config_.tsamp);
                // cout << std::setprecision(8) << std::fixed << headerfil.tstart << endl;
                sendtime = filbuffer_->gettime(ready-1);
                headerfil.tstart = get_mjd(sendtime.start_epoch, sendtime.start_second + 27 + sendtime.framet * config_.tsamp);
                // cout << std::setprecision(8) << std::fixed << headerfil.tstart << endl;
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
                    cout << ready - 1 << " buffer ready " << endl;
                filbuffer_ -> SendToRam(d_dedisp, ready, dedispstream_, (gulps_sent % 2));
                filbuffer_ -> SendToDisk((gulps_sent % 2), headerfil, config_.outdir);
                gulps_sent++;
                if ((int)(gulps_sent * dedispdispersedsamples_ * config_.tsamp) >= record_)
                    working_ = false;
            }   else {
                // perform the scaling
                filbuffer_->GetScaling(ready, dedispstream_, dmeans_, drstdevs_);
                cudaCheckError(cudaGetLastError());
                scaled_ = true;
                ready = 0;
                if (verbose_)
                    cout << "Scaling factors have been obtained" << endl;
            }

        } else {
            std::this_thread::yield();
        }
    }
}

// TODO: Change this function to be able to use config-specified port numbers
void GpuPool::ReceiveData(int portid, int recport)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + nostreams_ + (int)(ii / 3), &cpuset);
    int retaff = pthread_setaffinity_np(receive_threads[ii].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for receive thread on port " << 17000 + ii << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Receive thread on port " << 17000 + ii << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    const int pack_per_worker_buf = packperbuffer_ / nostreams_;
    int numbytes{0};
    short fpga{0};
    short bufidx{0};
    // this will always be an integer
    int frame{0};
    int ref_s{0};
    int packcount{0};
    int group{0};

    if (ii == 0) {
        unsigned char *temp_buf = receivebuffers_[0];
        numbytes = recvfrom(filedesc_[ii], receivebuffers_[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len);
        start_time.start_epoch = (int)(temp_buf[12] >> 2);
        start_time.start_second = (int)(temp_buf[3] | (temp_buf[2] << 8) | (temp_buf[1] << 16) | ((temp_buf[0] & 0x3f) << 24));
        beamno_ = (int)(temp_buf[23] | (temp_buf[22] << 8));
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    while (true) {
        if ((numbytes = recvfrom(filedesc_[ii], receivebuffers_[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << 17100 + ii << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        frame = (int)(receivebuffers_[ii][7] | (receivebuffers_[ii][6] << 8) | (receivebuffers_[ii][5] << 16) | (receivebuffers_[ii][4] << 24));
        if (frame == 0) {
            break;
        }
    }

    while(working_) {
        if ((numbytes = recvfrom(filedesc_[ii], receivebuffers_[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << 17100 + ii << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        ref_s = (int)(receivebuffers_[ii][3] | (receivebuffers_[ii][2] << 8) | (receivebuffers_[ii][1] << 16) | ((receivebuffers_[ii][0] & 0x3f) << 24));
        frame = (int)(receivebuffers_[ii][7] | (receivebuffers_[ii][6] << 8) | (receivebuffers_[ii][5] << 16) | (receivebuffers_[ii][4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frame = frame + (ref_s - start_time.start_second - 27) / 27 * 250000;

        // TODO: Add some mininal missing frames checks later anyway
        bufidx = ((int)(frame / accumulate_) % nostreams_) * pack_per_worker_buf;
        bufidx += (frame % accumulate_) * 48;
        frame_times[frame % (accumulate_ * nostreams_)] = frame;
        bufidx += fpga;
        std::copy(receivebuffers_[ii] + HEADER, receivebuffers_[ii] + BUFLEN, hinbuffer_ + (BUFLEN - HEADER) * bufidx);
        readybuffidx_[bufidx] = true;
    }
}
