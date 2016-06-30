#include <algorithm>
#include <bitset>
#include <iostream>
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
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "heimdall/pipeline.hpp"
#include "kernels.cuh"
#include "pdif.hpp"
#include "pool_multi.cuh"

using boost::asio::ip::udp;
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
#define PORTS 6

mutex cout_guard;

/* ########################################################
TODO: Too many copies - could I use move in certain places?
#########################################################*/

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

GPUpool::GPUpool(int id, config_s config) : gpuid(id),
                                        highest_frame(-1),
                                        highest_buf(0),
                                        batchsize(config.batch),
                                        fftpoint(config.fftsize),
                                        timeavg(config.timesavg),
                                        freqavg(config.freqavg),
                                        nostreams(config.streamno),
                                        npol(config.npol),
                                        d_rearrange_size(8 * config.batch * config.fftsize * config.timesavg),
                                        d_in_size(config.batch * config.fftsize * config.timesavg * config.npol),
                                        d_fft_size(config.batch * config.fftsize * config.timesavg * config.npol),
                                        d_power_size(config.batch * config.fftsize * config.timesavg),
                                        d_time_scrunch_size((config.fftsize - 5) * config.batch),
                                        d_freq_scrunch_size((config.fftsize - 5) * config.batch / config.freqavg),
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

    p_dedisp = unique_ptr<DedispPlan>(new DedispPlan(_config.filchans, _config.tsamp, _config.ftop, _config.foff, gpuid));
    p_mainbuffer = unique_ptr<Buffer<float>>(new Buffer<float>(gpuid));
    std::shared_ptr<boost::asio::io_service> iop(new boost::asio::io_service);
    ios = iop;

    // every thread will be associated with its own CUDA streams
    mystreams = new cudaStream_t[4];
    // each stream will have its own cuFFT plan
    myplans = new cufftHandle[4];

    int nkernels = 4;
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];
    // TODO: make a private const data memmber and put in the initializer list!!
    nchans = _config.nchans;

    // HASK: very quick solution to making the 21MHz work
    CUDAthreads[0] = 7;
    CUDAthreads[1] = 32 * 4 * 21 / 3;	// 2688 / 3 - need 3 blocks!
    CUDAthreads[2] = nchans;		// 21 - fine!
    CUDAthreads[3] = 21 * 27 / 9;	// 63 - fine!

    CUDAblocks[0] = 48;
    CUDAblocks[1] = 3;
    CUDAblocks[2] = 1;
    CUDAblocks[3] = 1;

    /* CUDAthreads[0] = fftpoint * timeavg * nchans;
    CUDAthreads[1] = nchans;
    CUDAthreads[2] = nchans * (fftpoint - 5) / freqavg;
    CUDAblocks[0] = 1;
    CUDAblocks[1] = 1;
    CUDAblocks[2] = 1;
    */
    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    cout << "Preparing the memory..." << endl;

    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());

    d_array2Dp = new cudaArray*[nostreams];
    texObj = new cudaTextureObject_t[nostreams];
    rdesc = new cudaResourceDesc[nostreams];
    tdesc = new cudaTextureDesc[3];

    for (int ii = 0; ii < nostreams; ii++) {
        cudaCheckError(cudaMallocArray(&(d_array2Dp[ii]), &cdesc, batchsize * fftpoint * timesavg));

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
    // this buffer takes two full bandwidths, 48 packets per bandwidth
    pack_per_buf = batchsize / 7 * 2;
    h_pol = new unsigned char[d_rearrange_size * 2];

    cudaCheckError(cudaHostAlloc((void**)&h_in, d_rearrange_size * nostreams * sizeof(unsigned char), cudaHostAllocDefault));
    cudaCheckError(cudaMalloc((void**)&d_in, d_in_size * nostreams * sizeof(cufftComplex)));
    cudaCheckError(cudaMalloc((void**)&d_fft, d_fft_size * nostreams * sizeof(cufftComplex)));
    // need to store all 4 Stoke parameters
    dv_power.resize(nostreams);
    dv_time_scrunch.resize(nostreams);
    dv_freq_scrunch.resize(nostreams);
    // TODO: make a private const data memmber and put in the initializer list!!
    stokes = _config.stokes;
    for (int ii = 0; ii < nostreams; ii++) {
        dv_power[ii].resize(d_power_size * stokes);
        dv_time_scrunch[ii].resize(d_time_scrunch_size * stokes);
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * stokes);
    }

    // STAGE: PREPARE THE DEDISPERSION
    // generate_dm_list(dm_start, dm_end, width, tol)
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials

    //dedisp.generate_dm_list(_config.dstart, _config.dend, 64.0f, 1.10f);
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
            cufftCheckError(cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftpoint, NULL, 1, fftpoint, CUFFT_C2C, batchsize * timeavg * npol));
            cufftCheckError(cufftSetStream(myplans[ii], mystreams[ii]));
            mythreads.push_back(thread(&GPUpool::worker, this, ii));
    }
    // dedispersion thread
    cudaCheckError(cudaStreamCreate(&mystreams[avt - 2]));
    mythreads.push_back(thread(&GPUpool::dedisp_thread, this, avt - 2));

    // STAGE: networking
    boost::asio::socket_base::reuse_address option(true);
    boost::asio::socket_base::receive_buffer_size option2(9000);


    for (int ii = 0; ii < 6; ii++) {
        sockets.push_back(udp::socket(*ios, udp::endpoint(boost::asio::ip::address::from_string("10.17.0.1"), 17100 + ii + 6 * gpuid)));
        sockets[ii].set_option(option);
        sockets[ii].set_option(option2);
        sender_endpoints.push_back(std::make_shared<udp::endpoint>());
    }

    mythreads.push_back(thread(&GPUpool::receive_thread, this));
    std::this_thread::sleep_for(std::chrono::seconds(1));
    //ios->run();
    sockets[0].async_receive_from(boost::asio::buffer(rec_buffer), *sender_endpoints[0], boost::bind(&GPUpool::receive_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, *sender_endpoints[0]));
    mythreads.push_back(thread([this]{ios->run();}));
    //mythreads[mythreads.size() - 1].join();

    cout << "Setting up networking..." << endl;

    start_time = 0;
    int netrv;
    addrinfo hints, *servinfo, tryme;
    char s[INET6_ADDRSTRLEN];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    sfds = new int[PORTS];

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

        for (tryme = servinfo; tryme != NULL; tryme->ai_next) {
            if((sfds[ii] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                cout << "Socke error\n";
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
        if(setsockopt(sfds[ii], SOL_SOCKET, SO_RECVBUF, &bufres, sizeof(int)) != 0) {
            cout << "Setsockopt error on port " << 17100 + ii << endl;
            cout << "Errno " << errno << endl;
        }
    }

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads.push_back(threads(&GPUpool::receive_thread, this, ii));

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads[ii].join();

    cout << "Done receiving..." << endl;
    cout.flush();
    // let the receive threads know they can work
}

GPUpool::~GPUpool(void)
{
    // TODO: clear the memory properly
    cout << "Calling destructor" << endl;
    cout.flush();
    for(int ii = 0; ii < mythreads.size(); ii++)
        mythreads[ii].join();

    for (int ii = 0; ii < nostreams; ii++) {
        cudaFreeArray(d_array2Dp[ii]);
    }
}

void GPUpool::worker(int stream)
{
    cudaSetDevice(gpuid);

    unsigned int skip = stream * d_in_size;

    int idx;
    int current_frame;

    float *pdv_power = thrust::raw_pointer_cast(dv_power[stream].data());
    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

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
            obs_time frame_time{start_time.start_epoch, start_time.start_second, current_frame};
            workermutex.unlock();
            index = index * d_rearrange_size;
            std::copy(h_pol + index,  h_pol + index + d_rearrange_size, h_in + skip);
            cudaCheckError(cudaMemcpyToArrayAsync(d_array2Dp[ii], 0, 0, h_in + skip, 8 * batchsize * fftpoint * timesavg, mystreams[stream]));
            rearrange<<<CUDAblock[0], CUDAthreads[0], mystreams[stream]>>>(texObj[ii], d_in + skip);
            cufftCheckError(cufftExecC2C(myplans[stream], d_in + skip, d_fft + skip, CUFFT_FORWARD));
            powerscale<<<CUDAblocks[1], CUDAthreads[1], 0, mystreams[stream]>>>(d_fft + skip, pdv_power, d_power_size);
            addtime<<<CUDAblocks[2], CUDAthreads[2], 0, mystreams[stream]>>>(pdv_power, pdv_time_scrunch, d_power_size, d_time_scrunch_size, timeavg);
            addchannel<<<CUDAblocks[3], CUDAthreads[3], 0, mystreams[stream]>>>(pdv_time_scrunch, pdv_freq_scrunch, d_time_scrunch_size, d_freq_scrunch_size, freqavg);
            // cudaPeekAtLastError does not reset the error to cudaSuccess like cudaGetLastError()
            // used to check for any possible errors in the kernel execution
            cudaCheckError(cudaPeekAtLastError());
            cudaThreadSynchronize();

            p_mainbuffer->write(pdv_freq_scrunch, frame_time, d_freq_scrunch_size, mystreams[stream]);

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
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));
    int numbytes{0};
    short fpga{0};
    short bufidx{0};
    int frame{0};
    int ref_s;
    // I want this thread to worry only about saving the data
    // TODO: make worker thread worry about picking the data up
    if (ii == 0) {
        unsigned char *temp_buf = rec_bufs[0];
        numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1);
        start_time.start_epoch = (int)(temp_buf[12] >> 2);
        start_time.start_second = (int)(temp_buf[3] | (temp_buf[2] << 8) | (temp_buf[1] << 16) | ((temp_buf[0] & 0x3f) << 24));
    }
    while(working) {
        if ((numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout << "Error of recvfrom on port " << 17100 + ii << endl;
            // possible race condition here
            cout << "Errno " << errno << endl;
        }
        ref_s = (int)(rec_bufs[ii][3] | (rec_bufs[ii][2] << 8) | (rec_bufs[1]) << 16) | ((rec_bufs[ii][0] & 0x3f) << 24));
        frame = (int)(rec_bufs[ii][7] | (rec_bufs[ii][6] << 8) | (rec_bufs[ii][5] << 16) | (rec_bufs[ii][4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 8 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frame = frame + (ref_s - start_time.start_second) / 27 * 250000;

        // looking at how much stuff we are not missing - remove a lot of checking for now
        // TODO: add some mininal checks later anyway

        bufidx = (frame % 2) * pack_per_buf + fpga;
        std::copy(rec_bufs[ii] + HEADER, rec_bufs[ii] + BUFLEN, h_pol + BUFLEN * bufidx);
        buffer_ready[(int)(bufidx / (pack_per_buf / 2)] = true;

        buffermutex.lock();
        if(buffer_ready[0] && bufidx >= 3 * (pack_per_buf / 4) ) {
            workermutex.lock();
                worker_ready[0] = true;
                worker_frame[0] = frame - 1;
            workermutex.unlock();
            buffer_ready[0] = false;
        } else if (buffer_ready[1] && bufidx >= (pack_per_buf /4)) {
            workermutex.lock();
                worker_ready[1] = true;
                worker_frame[1] = frame - 1;
            workermutex.unlock();
            buffer_ready[0] = false;
        }
        buffermutex.unlock();
    }
}
