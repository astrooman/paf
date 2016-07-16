#include <algorithm>
#include <bitset>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cufft.h>
#include <cuda.h>
#include <pthread.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "heimdall/pipeline.hpp"
#include "kernels.cuh"
#include "paf_metadata.hpp"
#include "pdif.hpp"
#include "pool_multi.cuh"

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
using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::unique_ptr;
using std::vector;

#define BYTES_PER_WORD 8
#define HEADER 64
#define WORDS_PER_PACKET 896
#define BUFLEN 7168 + 64
#define PORTS 8

mutex cout_guard;

/* ########################################################
TODO: Too many copies - could I use move in certain places?
#########################################################*/

/*##############################################
IMPORTANT: from what I seen in the system files:
eth3, gpu0, gpu1 - NUMA node 0, CPUs 0-7
eth2, gpu2, gpu3 - NUMA node 1, CPUs 8-15
##############################################*/

Oberpool::Oberpool(config_s config) : ngpus(config.ngpus)
{

    for (int ii = 0; ii < ngpus; ii++) {
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii, config)));
    }

    for (int ii = 0; ii < ngpus; ii++) {
        threadvector.push_back(thread(&GPUpool::execute, std::move(gpuvector[ii])));
    }
}

Oberpool::~Oberpool(void)
{
    for (int ii = 0; ii < ngpus; ii++) {
        threadvector[ii].join();
    }
}

GPUpool::GPUpool(int id, config_s config) : accumulate(config.accumulate),
                                        gpuid(2),
                                        highest_frame(-1),
                                        highest_buf(0),
                                        batchsize(config.batch),
                                        fftpoint(config.fftsize),
                                        timeavg(config.timesavg),
                                        freqavg(config.freqavg),
                                        nostreams(config.streamno),
                                        npol(config.npol),
                                        d_rearrange_size(8 * config.batch * config.fftsize * config.timesavg * config.accumulate),
                                        d_in_size(config.batch * config.fftsize * config.timesavg * config.npol * config.accumulate),
                                        d_fft_size(config.batch * config.fftsize * config.timesavg * config.npol * config.accumulate),
                                        d_time_scrunch_size((config.fftsize - 5) * config.batch * config.accumulate),
                                        d_freq_scrunch_size((config.fftsize - 5) * config.batch  * config.accumulate / config.freqavg),
                                        gulps_sent(0),
                                        gulps_processed(0),
                                        working(true),
					                    packcount(0)

{
    cout << "New pool" << endl;

    avt = min(nostreams + 2, thread::hardware_concurrency());

    _config = config;

    if (config.verbose) {
        cout_guard.lock();
        cout << "Starting GPU pool " << gpuid << endl;
	    cout.flush();
        cout_guard.unlock();
    }
}

void GPUpool::execute(void)
{
    cudaCheckError(cudaSetDevice(gpuid));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(8, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0)
        cout << "Error setting thread affinity for the GPU pool " << gpuid << endl;

    cout << "GPU pool for device " << gpuid << " running on CPU " << sched_getcpu() << endl;

    p_dedisp = unique_ptr<DedispPlan>(new DedispPlan(_config.filchans, _config.tsamp, _config.ftop, _config.foff, gpuid));
    p_mainbuffer = unique_ptr<Buffer<float>>(new Buffer<float>(gpuid));

    frame_times = new int[2 * accumulate];
    // every thread will be associated with its own CUDA streams
    mystreams = new cudaStream_t[avt];
    // each worker stream will have its own cuFFT plan
    myplans = new cufftHandle[nostreams];

    int nkernels = 4;
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];
    // TODO: make a private const data memmber and put in the initializer list!!
    nchans = _config.nchans;

    CUDAthreads[0] = 7;
    CUDAthreads[1] = fftpoint * timeavg * batchsize / 42;
    CUDAthreads[2] = nchans;		// 21 - fine!
    CUDAthreads[3] = batchsize * 27 / freqavg;	// 63 - fine!

    CUDAblocks[0] = 48;
    CUDAblocks[1] = 42;
    CUDAblocks[2] = 1;
    CUDAblocks[3] = 1;

    for (int ii = 0; ii < nkernels; ii++)
        cout << "Kernel " << ii << ": block - " << CUDAblocks[ii] << ", thread - " << CUDAthreads[ii] << endl;

    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    cout << "Preparing the memory..." << endl;

    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());

    d_array2Dp = new cudaArray*[nostreams];
    texObj = new cudaTextureObject_t[nostreams];
    rdesc = new cudaResourceDesc[nostreams];
    tdesc = new cudaTextureDesc[nostreams];

    for (int ii = 0; ii < nostreams; ii++) {
        cudaCheckError(cudaMallocArray(&(d_array2Dp[ii]), &cdesc, 7, (batchsize  / 7) * fftpoint * timeavg * accumulate));

        memset(&(rdesc[ii]), 0, sizeof(cudaResourceDesc));
        rdesc[ii].resType = cudaResourceTypeArray;
        rdesc[ii].res.array.array = d_array2Dp[ii];

        memset(&(tdesc[ii]), 0, sizeof(cudaTextureDesc));
        tdesc[ii].addressMode[0] = cudaAddressModeClamp;
        tdesc[ii].addressMode[1] = cudaAddressModeClamp;
        tdesc[ii].filterMode = cudaFilterModePoint;
        tdesc[ii].readMode = cudaReadModeElementType;

        texObj[ii] = 0;
        cudaCheckError(cudaCreateTextureObject(&(texObj[ii]), &(rdesc[ii]), &(tdesc[ii]), NULL));
    }

    // it has to be an array and I can't do anything about that
    sizes[0] = (int)fftpoint;

    pack_per_buf = batchsize / 7 * accumulate * 2;
    h_pol = new unsigned char[d_rearrange_size * 2];

    cudaCheckError(cudaHostAlloc((void**)&h_in, d_rearrange_size * nostreams * sizeof(unsigned char), cudaHostAllocDefault));
    cudaCheckError(cudaMalloc((void**)&d_in, d_in_size * nostreams * sizeof(cufftComplex)));
    cudaCheckError(cudaMalloc((void**)&d_fft, d_fft_size * nostreams * sizeof(cufftComplex)));
    // need to store all 4 Stoke parameters
    dv_time_scrunch.resize(nostreams);
    dv_freq_scrunch.resize(nostreams);
    // TODO: make a private const data memmber and put in the initializer list!!
    stokes = _config.stokes;
    for (int ii = 0; ii < nostreams; ii++) {
        dv_time_scrunch[ii].resize(d_time_scrunch_size * stokes);
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * stokes);
    }

    // STAGE: PREPARE THE DEDISPERSION
    // generate_dm_list(dm_start, dm_end, width, tol)
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials
    p_dedisp->generate_dm_list(_config.dstart, _config.dend, 64.0f, 1.10f);
    // this is the number of time sample - each timesample will have config.filchans frequencies
    dedisp_totsamples = (size_t)_config.gulp + 0; //p_dedisp->get_max_delay();
    dedisp_buffno = (dedisp_totsamples - 1) / _config.gulp + 1;
    dedisp_buffsize = dedisp_buffno * _config.gulp + 0; //p_dedisp->get_max_delay();
    cout << "Total buffer size: " << dedisp_buffsize << endl;
    // can this method be simplified?
    p_mainbuffer->allocate(dedisp_buffno, 0, _config.gulp, dedisp_buffsize, _config.filchans, stokes);
    buffer_ready[0] = false;
    buffer_ready[1] = false;
    worker_ready[0] = false;
    worker_ready[1] = false;
    p_dedisp->set_killmask(&_config.killmask[0]);
    // everything should be ready for dedispersion after this point

    // STAGE: PREPARE THE SINGLE PULSE SEARCH
    cout << "Setting up dedispersion and single pulse search..." << endl;
    set_search_params(params, _config);
    //commented out for the filterbank dump mode
    //hd_create_pipeline(&pipeline, params);
    // everything should be ready for single pulse search after this point

    // STAGE: start processing
    // FFT threads
    for (int ii = 0; ii < nostreams; ii++) {
            cudaCheckError(cudaStreamCreate(&mystreams[ii]));
            // TODO: add separate error checking for cufft functions
            cufftCheckError(cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftpoint, NULL, 1, fftpoint, CUFFT_C2C, batchsize * timeavg * npol * accumulate));
            cufftCheckError(cufftSetStream(myplans[ii], mystreams[ii]));
            mythreads.push_back(thread(&GPUpool::worker, this, ii));
    }
    // dedispersion thread
    cudaCheckError(cudaStreamCreate(&mystreams[avt - 2]));
    mythreads.push_back(thread(&GPUpool::dedisp_thread, this, avt - 2));

    // STAGE: networking

    cout << "Setting up networking..." << endl;

    memset(&start_time, 0, sizeof(start_time)) ;
    int netrv;
    addrinfo hints, *servinfo, *tryme;
    char s[INET6_ADDRSTRLEN];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    sfds = new int[PORTS];

    rec_bufs = new unsigned char*[PORTS];
    for (int ii = 0; ii < PORTS; ii++)
        rec_bufs[ii] = new unsigned char[BUFLEN];

    std::ostringstream oss;
    std::string strport;

    // all the magic happens here
    for (int ii = 0; ii < PORTS; ii++) {

        oss.str("");
        oss << 17100 + ii;
        strport = oss.str();
        cout << "Binding to port " << strport << "..." << endl;
        cout.flush();

        if((netrv = getaddrinfo("10.17.0.1", strport.c_str(), &hints, &servinfo)) != 0) {
            cout << "getaddrinfo() error: " << gai_strerror(netrv) << endl;
        }

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((sfds[ii] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                cout << "Socket error\n";
                continue;
            }

            if(bind(sfds[ii], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(sfds[ii]);
                cout << "Bind error\n";
                continue;
            }
            break;
        }

        if (tryme == NULL) {
            cout << "Failed to bind to the socket\n";
        }
    }

    int bufres{9000};

    for (int ii = 0; ii < PORTS; ii++) {
        if(setsockopt(sfds[ii], SOL_SOCKET, SO_RCVBUF, &bufres, sizeof(int)) != 0) {
            cout << "Setsockopt error on port " << 17100 + ii << endl;
            cout << "Errno " << errno << endl;
        }
    }

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads.push_back(thread(&GPUpool::receive_thread, this, ii));

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads[ii].join();

    // TODO: this thread does nothing at this point so might as well make it listen to metadata

    cout << "Setting up metadata logger..." << endl;

    int metabytes, sock_meta;
    addrinfo hints_meta, *servinfo_meta, *tryme_meta;
    memset(&hints_meta, 0, sizeof(hints_meta));
    hints_meta.ai_family = AF_INET;
    hints_meta.ai_socktype = SOCK_DGRAM;
    hints_meta.ai_flags = AI_PASSIVE;
    sockaddr_storage meta_addr;
    memset(&meta_addr, 0, sizeof(meta_addr));
    socklen_t meta_len;
    memset(&meta_len, 0, sizeof(meta_len));

    if ((netrv = getaddrinfo("130.155.182.74", "26666", &hints_meta, &servinfo_meta)) != 0) {
        cout << "gettaddrinfo() error on metadata socket 26666" << endl;

        for (tryme_meta = servinfo_meta; tryme_meta != NULL; tryme_meta=tryme_meta->ai_next) {
            if ((sock_meta = socket(tryme_meta->ai_family, tryme_meta->ai_socktype, tryme_meta->ai_protocol)) == -1) {
                cout << "Metadata socket error\n";
                continue;
            }
            if (bind(sock_meta, tryme_meta->ai_addr, tryme_meta->ai_addrlen) == -1) {
                cout << "Metadata bind error\n";
                continue;
            }
            break;
        }

        if (tryme_meta == NULL) {
            cout << "Failed to bind to the metadata socket\n";
        }
    }

    metadata paf_meta;
    std::fstream metalog("metadata_log.dat", std::ios_base::out | std::ios_base::trunc);

    char *metabuffer = new char[4096];

/*    if (metalog) {
        while(working) {
            metabytes = recvfrom(sock_meta, metabuffer, 4096, 0, (struct sockaddr*)&meta_addr, &meta_len);

            string metastr(metabuffer);
            paf_meta.getMetaData(metastr, 0);
            cout << paf_meta.timestamp << "\t";
            cout << paf_meta.beam_num << "\t";
            cout << paf_meta.beam_ra << "\t";
            cout << paf_meta.beam_dec << "\t";
            cout << paf_meta.target_name << endl;
            cout.flush();

            metalog << paf_meta.timestamp << "\t";
            metalog << paf_meta.beam_num << "\t";
            metalog << paf_meta.beam_ra << "\t";
            metalog << paf_meta.beam_dec << "\t";
            metalog << paf_meta.target_name << endl << endl;
        }

        metalog.close();
    } else {
        cout << "Metadata log file error!!" << endl;
    }
*/
    delete [] metabuffer;

    cout << "Done receiving..." << endl;
    cout.flush();
    // let the receive threads know they can work
}

GPUpool::~GPUpool(void)
{
    // TODO: clear the memory properly
    cout << "Calling destructor" << endl;
    cout.flush();
    working = false;
    for(int ii = 0; ii < mythreads.size(); ii++)
        mythreads[ii].join();

    for (int ii = 0; ii < nostreams; ii++) {
        cudaFreeArray(d_array2Dp[ii]);
    }
}

void GPUpool::worker(int stream)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(10 + stream, &cpuset);
    int retaff = pthread_setaffinity_np(mythreads[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0)
        cout<< "Error setting thread affinity for stream " << stream << endl;

    printmutex.lock();
    cout << "Starting worker " << gpuid << ":" << stream << " on CPU " << sched_getcpu() << endl;
    cout.flush();
    printmutex.unlock();

    cudaSetDevice(gpuid);

    unsigned int skip = stream * d_in_size;

    int idx;
    unsigned int current_frame;

    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

    float **p_fil = p_mainbuffer->get_pfil();
    float **pd_fil;
    cudaMalloc((void**)&pd_fil, stokes * sizeof(float *));
    cudaMemcpy(pd_fil, p_fil, stokes * sizeof(float *), cudaMemcpyHostToDevice);

    while(working) {
        unsigned int index{0};
        workermutex.lock();
        if(worker_ready[0] || worker_ready[1]) {
            if(worker_ready[0]) {
                index = 0;
                worker_ready[0] = false;
                current_frame = worker_frame[0];
            } else {
                index = 1;
                worker_ready[1] = false;
                current_frame = worker_frame[1];
            }
            //cout << "Got some stuff!!" << endl;
            //cout.flush();
            obs_time frame_time{start_time.start_epoch, start_time.start_second, current_frame};
            workermutex.unlock();
            index = index * d_rearrange_size;
            std::copy(h_pol + index,  h_pol + index + d_rearrange_size, h_in + stream * d_rearrange_size);;
            for (int frameidx = 0; frameidx < acculumate; frameidx++) {
                if (thread_frames[frameidx] != 0) {
                    current_frame = frame_times[frameidx];
                    break;
                }
            }
            obs_time frame_time{start_time.start_epoch, start_time.start_second, current_frame};
            cudaCheckError(cudaMemcpyToArrayAsync(d_array2Dp[stream], 0, 0, h_in + stream * d_rearrange_size, d_rearrange_size, cudaMemcpyHostToDevice, mystreams[stream]));
            rearrange2<<<CUDAblocks[0], CUDAthreads[0], 0, mystreams[stream]>>>(texObj[stream], d_in + skip, accumulate);
            cufftCheckError(cufftExecC2C(myplans[stream], d_in + skip, d_fft + skip, CUFFT_FORWARD));
            powertime2<<<48, 27, 0, mystreams[stream]>>>(d_fft + skip, pdv_time_scrunch, d_time_scrunch_size, timeavg, accumulate);
            addchannel2<<<CUDAblocks[3], CUDAthreads[3], 0, mystreams[stream]>>>(pdv_time_scrunch, pd_fil, (short)_config.filchans, _config.gulp, dedisp_buffsize, dedisp_buffno, d_time_scrunch_size, freqavg, current_frame, accumulate);
            // used to check for any possible errors in the kernel execution
            cudaCheckError(cudaGetLastError());
            cudaThreadSynchronize();
            p_mainbuffer->update(frame_time);

        } else {
            workermutex.unlock();
            std::this_thread::yield();
        }
    }
}

void GPUpool::dedisp_thread(int dstream)
{

    cudaCheckError(cudaSetDevice(gpuid));
    while(working) {
        int ready = p_mainbuffer->ready();
        if (ready) {
            header_f headerfil;
            headerfil.raw_file = "tastytastytest";
            headerfil.source_name = "J1641-45";
            headerfil.az = 0.0;
            headerfil.dec = 0.0;
            headerfil.fch1 = _config.ftop;
            headerfil.foff = _config.foff;
            headerfil.ra = 0.0;
            headerfil.rdm = 0.0;
            headerfil.tsamp = _config.tsamp;
            headerfil.tstart = 0.0;
            headerfil.za = 0.0;
            headerfil.data_type = 1;
            headerfil.ibeam = 0;
            headerfil.machine_id = 2;
            headerfil.nbeams = 1;
            headerfil.nbits = 32;
            headerfil.nchans = _config.filchans;
            headerfil.nifs = 1;
            headerfil.telescope_id = 2;

            p_mainbuffer->send(d_dedisp, ready, mystreams[dstream], (gulps_sent % 2));
            p_mainbuffer->dump((gulps_sent % 2), headerfil);
            gulps_sent++;
        } else {
            std::this_thread::yield();
        }
    }
}

void GPUpool::receive_thread(int ii)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ii, &cpuset);
    int retaff = pthread_setaffinity_np(receive_threads[ii].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0)
        cout << "Error setting thread affinity for receive thread on port " << 17000 + ii << endl;
    printmutex.lock();
    cout << "Receive thread on port " << 17000 + ii << " running on CPU " << sched_getcpu() << endl;
    printmutex.unlock();

    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    int numbytes{0};
    short fpga{0};
    short bufidx{0};
    // this will always be an integer
    short intofirst = (short)(0.5 * (double)pack_per_buf + 0.5 * (double)pack_per_buf / (double)accumulate);
    short intosecond = (short)(0.5 * (double)pack_per_buf / (double)accumulate);
    int frame{0};
    int ref_s;
    int packcount;
    // I want this thread to worry only about saving the data
    // TODO: make worker thread worry about picking the data up
    if (ii == 0) {
        unsigned char *temp_buf = rec_bufs[0];
        numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len);
        start_time.start_epoch = (int)(temp_buf[12] >> 2);
        start_time.start_second = (int)(temp_buf[3] | (temp_buf[2] << 8) | (temp_buf[1] << 16) | ((temp_buf[0] & 0x3f) << 24));
    }
    while(working) {
        if ((numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout << "Error of recvfrom on port " << 17100 + ii << endl;
            // possible race condition here
            cout << "Errno " << errno << endl;
        }
        ref_s = (int)(rec_bufs[ii][3] | (rec_bufs[ii][2] << 8) | (rec_bufs[ii][1] << 16) | ((rec_bufs[ii][0] & 0x3f) << 24));
        frame = (int)(rec_bufs[ii][7] | (rec_bufs[ii][6] << 8) | (rec_bufs[ii][5] << 16) | (rec_bufs[ii][4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 8 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frame = frame + (ref_s - start_time.start_second) / 27 * 250000;

        // looking at how much stuff we are not missing - remove a lot of checking for now
        // TODO: add some mininal checks later anyway

        // which half of the buffer to put the data in
        bufidx = ((int)(frame / accumulate) % 2) * (pack_per_buf / 2);
        // frame position in the half
        bufidx += (frame % accumulate) * 48;
        frame_times[frame % (2 * accumulate)] = frame;
        // frequency chunk in the frame
        bufidx += fpgaid;
        std::copy(rec_bufs[ii] + HEADER, rec_bufs[ii] + BUFLEN, h_pol + BUFLEN * bufidx);
        buffer_ready[(int)(bufidx / (pack_per_buf / 2))] = true;

        buffermutex.lock();
        if(buffer_ready[0] && bufidx >= intosecond ) {
            workermutex.lock();
                worker_ready[0] = true;
                worker_frame[0] = frame - 1;
            workermutex.unlock();
            buffer_ready[0] = false;
        } else if (buffer_ready[1] && bufidx >= intofirst) {
            workermutex.lock();
                worker_ready[1] = true;
                worker_frame[1] = frame - 1;
            workermutex.unlock();
            buffer_ready[0] = false;
        }
        buffermutex.unlock();
        //packcount++;
    }
}
