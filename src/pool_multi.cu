#include <bitset>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
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
                                        batchsize(config.batch),
                                        fftpoint(config.fftsize),
                                        timeavg(config.timesavg),
                                        freqavg(config.freqavg),
                                        nostreams(config.streamno),
                                        npol(config.npol),
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

    int nkernels = 3;
    // [0] - powerscale() kernel, [1] - addtime() kernel, [2] - addchannel() kernel
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];
    // [0] - powerscale() kernel, [1] - addtime() kernel, [2] - addchannel() kernel
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];
    // TODO: make a private const data memmber and put in the initializer list!!
    nchans = _config.nchans;

    // HASK: very quick solution to making the 21MHz work
    CUDAthreads[0] = 32 * 4 * 21 / 3;	// 2688 / 3 - need 3 blocks!
    CUDAthreads[1] = nchans;		// 21 - fine!
    CUDAthreads[2] = 21 * 27 / 9;	// 63 - fine!

    CUDAblocks[0] = 3;
    CUDAblocks[1] = 1;
    CUDAblocks[2] = 1;

    /* CUDAthreads[0] = fftpoint * timeavg * nchans;
    CUDAthreads[1] = nchans;
    CUDAthreads[2] = nchans * (fftpoint - 5) / freqavg;
    CUDAblocks[0] = 1;
    CUDAblocks[1] = 1;
    CUDAblocks[2] = 1;
    */
    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    // it has to be an array and I can't do anything about that
    sizes[0] = (int)fftpoint;
    // this buffer takes two full bandwidths, 48 packets per bandwidth
    // TEST
    //pack_per_buf = 12;
    pack_per_buf = 4;
    h_pol = new cufftComplex[d_in_size * 2];

    cudaCheckError(cudaHostAlloc((void**)&h_in, d_in_size * nostreams * sizeof(cufftComplex), cudaHostAllocDefault));
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
    mythreads.push_back(thread([this]{ios->run();}));
    //mythreads[mythreads.size() - 1].join();
}

GPUpool::~GPUpool(void)
{
    // TODO: clear the memory properly
    cout << "Calling destructor" << endl;
    cout.flush();
    for(int ii = 0; ii < mythreads.size(); ii++)
        mythreads[ii].join();
}

void GPUpool::worker(int stream)
{
    cudaSetDevice(gpuid);

    unsigned int skip = stream * d_in_size;

    float *pdv_power = thrust::raw_pointer_cast(dv_power[stream].data());
    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

    while(working) {
        unsigned int index{0};
        datamutex.lock();
        if(!mydata.empty()) {
            std::copy((mydata.front()).first.begin(), (mydata.front()).first.end(), h_in + skip);
            obs_time framte_time = mydata.front().second;
            mydata.pop();
            datamutex.unlock();
            cudaCheckError(cudaMemcpyAsync(d_in + skip, h_in + skip, d_in_size * sizeof(cufftComplex), cudaMemcpyHostToDevice, mystreams[stream]));
            cufftCheckError(cufftExecC2C(myplans[stream], d_in + skip, d_fft + skip, CUFFT_FORWARD));
            powerscale<<<CUDAblocks[0], CUDAthreads[0], 0, mystreams[stream]>>>(d_fft + skip, pdv_power, d_power_size);
            addtime<<<CUDAblocks[1], CUDAthreads[1], 0, mystreams[stream]>>>(pdv_power, pdv_time_scrunch, d_power_size, d_time_scrunch_size, timeavg);
            addchannel<<<CUDAblocks[2], CUDAthreads[2], 0, mystreams[stream]>>>(pdv_time_scrunch, pdv_freq_scrunch, d_time_scrunch_size, d_freq_scrunch_size, freqavg);
            // cudaPeekAtLastError does not reset the error to cudaSuccess like cudaGetLastError()
            // used to check for any possible errors in the kernel execution
            cudaCheckError(cudaPeekAtLastError());
            cudaThreadSynchronize();

            p_mainbuffer->write(pdv_freq_scrunch, framte_time, d_freq_scrunch_size, mystreams[stream]);

        } else {
            datamutex.unlock();
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

void GPUpool::receive_thread(void)
{
    sockets[0].async_receive_from(boost::asio::buffer(rec_buffer), *sender_endpoints[0], boost::bind(&GPUpool::receive_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, *sender_endpoints[0]));
}

void GPUpool::receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint endpoint)
{
    header_s head;

    // I don't want to call this every single time - preferably only once
    get_header(rec_buffer.data(), head);
    static obs_time start_time{head.epoch, head.ref_s};
    // this is ugly, but I don't have a better solution at the moment
    int long_ip = boost::asio::ip::address_v4::from_string((endpoint.address()).to_string()).to_ulong();
    int fpga = ((int)((long_ip >> 8) & 0xff) - 1) * 8 + ((int)(long_ip & 0xff) - 1) / 2;

    get_data(rec_buffer.data(), fpga, start_time, head);
    packcount++;
    if (packcount > 0)
	receive_thread();
}

void GPUpool::get_data(unsigned char* data, int fpga_id, obs_time start_time, header_s head)
{
    // REMEMBER - d_in_size is the size of the single buffer (2 polarisations, 336 channels, 128 time samples)
    unsigned int idx = 0;
    unsigned int idx2 = 0;

    // there are 250,000 frames per 27s period
    int frame = head.frame_no + (head.ref_s - start_time.start_second) / 27 * 250000;
    // even-numbered frames saved in the first half of the buffer
    // odd-numbered frames saved in the second half of the buffer
    // for this test use fpga_id % 2, because at the end of the day I only care about their relative positions
    int bufidx = fpga_id % 2 + (frame % 2) * 2;                                    // received packet number in the current buffer
    size_t buftot = (size_t)frame * (size_t)(pack_per_buf / 2) + (size_t)fpga_id;

    if ((buftot - highest_buf) > 1) {
        cout << "Missed " << bufidx - highest_buf - 1 << " packets " << endl;
        cout.flush();
    }

    if (frame > highest_frame) {
        if ((frame - highest_frame) > 1) {
            cout << "Missed " << frame - highest_frame - 1 << "  whole frames" << endl;
            cout.flush();
            // TODO: sort this out - I don't want to have if statement all the time, when this will be relevant once only
            if(highest_frame != -1) {
                add_data(h_pol + d_in_size  * (highest_frame % 2) , {start_time.start_epoch, start_time.start_second, highest_frame});
                buffer_ready[highest_frame % 2] = false;
            }
        }
        highest_frame = frame;
    }

    if (buffer_ready[0] && (bufidx >= (3 * pack_per_buf / 4))) {                     // if 1 sample or more into the second buffer - send first one
        add_data(h_pol, {start_time.start_epoch, start_time.start_second, highest_frame - 1});
        buffer_ready[0] = false;
    } else if(buffer_ready[1] && (bufidx >= pack_per_buf / 4)) {        // if 1 sample or more into the first buffer and second buffer has been filled - send second one
        add_data(h_pol + d_in_size, {start_time.start_epoch, start_time.start_second, highest_frame - 1});
        buffer_ready[1] = false;
    }

    int startidx = ((int)(bufidx / pack_per_buf) * pack_per_buf + bufidx) * WORDS_PER_PACKET;       // starting index for the packet in the buffer
										// used to skip second polarisation data

    // I don't care about the data that is more than half a single buffer late
    if ((highest_buf - buftot) < pack_per_buf / 4 ) {
        #pragma unroll
        for (int chan = 0; chan < 7; chan++) {
            for (int sample = 0; sample < 128; sample++) {
                idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
                idx2 = chan * 128 + sample + startidx;        // get the position in the buffer
                h_pol[idx2].x = static_cast<float>(static_cast<short>(data[HEADER + idx + 7] | (data[HEADER + idx + 6] << 8)));
                h_pol[idx2].y = static_cast<float>(static_cast<short>(data[HEADER + idx + 5] | (data[HEADER + idx + 4] << 8)));
                h_pol[idx2 + d_in_size / 2].x = static_cast<float>(static_cast<short>(data[HEADER + idx + 3] | (data[HEADER + idx + 2] << 8)));
                h_pol[idx2 + d_in_size / 2].y = static_cast<float>(static_cast<short>(data[HEADER + idx + 1] | (data[HEADER + idx + 0] << 8)));
            }
        }
        buffer_ready[(int)(bufidx / pack_per_buf)] = true;
    }
}

void GPUpool::add_data(cufftComplex *buffer, obs_time frame_time)
{
    std::lock_guard<mutex> addguard(datamutex);
    // TODO: is it possible to simplify this messy line?
    mydata.push(pair<vector<cufftComplex>, obs_time>(vector<cufftComplex>(buffer, buffer + d_in_size), frame_time));
    //cout << "Pushed the data to the queue of current length of " << mydata.size() << endl;
}
