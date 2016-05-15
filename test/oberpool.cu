#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "pdif.hpp"

using boost::asio::ip::udp;
using std::cout;
using std::endl;
using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::vector;
using std::unique_ptr;

std::mutex cout_guard;

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort)
{
    // index will tell which 1MHz channel we are taking care or
    // use 1 thread per 1MHz channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int ch = 0; ch < 27; ch++) {
        for (int t = 0; t < factort; t++) {
            out[idx * 27 + ch] += in[idx * 128 + ch + t * 32];
            out[idx * 27 + ch + jumpout] += in[idx * 128 + ch + t * 32 + jumpin];
            out[idx * 27 + ch + 2 * jumpout] += in[idx * 128 + ch + t * 32 + 2 * jumpin];
            out[idx * 27 + ch + 3 * jumpout] += in[idx * 128 + ch + t * 32 + 3 * jumpin];
        }
    }

}

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int ch = 0; ch < factorc; ch++) {
        out[idx] += in[idx * factorc + ch];
        out[idx + jumpout] += in[idx * factorc + ch + jumpin];
        out[idx + 2 * jumpout] += in[idx * factorc + ch + 2 * jumpin];
        out[idx + 3 * jumpout] += in[idx * factorc + ch + 3 * jumpin];
    }

}

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump)
{
    //printf("In the power kernel\n");
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced, jump to the B polarisation data - can cause some slowing down
	int idx2 = idx1 + jump;
    // these calculations assume polarisation is recorded in x,y base
    // i think the if statement is unnecessary as the number of threads for this
    // kernel 0s fftpoint * timeavg * nchans, which is exactly the size of the output array
    if (idx1 < jump) {      // half of the input data
        float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
        float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
        out[idx1] = (power1 + power2); // I; what was this doing here? / 2.0;
        out[idx1 + jump] = (power1 - power2); // Q
        out[idx1 + 2 * jump] = 2 * in[idx1].x * in[idx2].x + 2 * in[idx1].y * in[idx2].y; // U
        out[idx1 + 3 * jump] = 2 * in[idx1].x * in[idx2].y - 2 * in[idx1].y * in[idx2].x; // V
    }
}

class GPUpool;

class Oberpool {

    public:
        Oberpool(void);
        Oberpool(const Oberpool &inpool) = delete;
        Oberpool& operator=(const Oberpool &inpool) = delete;
        Oberpool(Oberpool &&inpool) = delete;
        Oberpool& operator=(Oberpool &&inpool) = delete;
        ~Oberpool(void);
    private:
        //vector<GPUpool> gpuvector;
        //vector<GPUpool*> gpuvector;
        vector<unique_ptr<GPUpool>> gpuvector;
        vector<thread> threadvector;
};

class GPUpool {

    public:
        GPUpool(void);
        GPUpool(int ii);
        GPUpool(const GPUpool &inpool) = delete;
        GPUpool& operator=(const GPUpool &inpool) = delete;
        GPUpool(GPUpool &&inpool) { cout << "Move constructor " << endl; id = inpool.id; inpool.id = 0;};
        GPUpool& operator=(GPUpool &&inpool) = delete;
        ~GPUpool(void);
        void add_data(cufftComplex *buffer, obs_time frame_time);
        void execute(void);
        void get_data(unsigned char* data, int frame, obs_time start_time);
	void receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint &endpoint);
        void receive_thread(void);
        void worker(int ii);
    private:
        Buffer<float> mainbuffer;
        std::shared_ptr<boost::asio::io_service> ios;
        vector<udp::socket> sockets;
        boost::array<unsigned char, 7168 + 64> rec_buffer;
	udp::endpoint sender_endpoint;
        cufftComplex *h_in;
        cufftComplex *h_pol;
        cufftComplex *d_in;
        cufftComplex *d_fft;
	double *buffer;
        int batchsize = 336;
        int packetcount = 0;
        int fftpoint = 32;
        int highest_frame;
        int d_fft_size;
        int d_in_size;
        int d_power_size;
        int d_time_scrunch_size;
        int d_freq_scrunch_size;
        int nstreams; 
        int id;
        int sizes[1] = {32};
        int stokes;
        mutex datamutex;
        unsigned int *CUDAthreads;
        unsigned int *CUDAblocks;
	queue<pair<vector<cufftComplex>, obs_time>> mydata;
        vector<thrust::device_vector<float>> dv_power;
        vector<thrust::device_vector<float>> dv_time_scrunch;
        vector<thrust::device_vector<float>> dv_freq_scrunch;
        vector<thread> mythreads;
        cudaStream_t *mystreams;
        cufftHandle *myplans;
        size_t dedisp_buffsize;
        size_t dedisp_totsamples;
        unsigned int dedisp_buffno;
};

Oberpool::Oberpool(void)
{
    cout << "Hello from Oberpool constructor!!" << endl;

    for (int ii = 0; ii < 1; ii++)
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii)));

    for (int ii = 0; ii < 1; ii++)
        threadvector.push_back(thread(&GPUpool::execute, std::move(gpuvector[ii])));

}

Oberpool::~Oberpool(void)
{
    cout_guard.lock();
    cout << "Bye from the Oberpool destructor!!" << endl;
    cout_guard.unlock();
    for (int ii = 0; ii < threadvector.size(); ii++)
        threadvector[ii].join();

    cout << "Now properly bye from the Oberpool destructor!!" << endl;
}

GPUpool::GPUpool(void) : d_in_size(7 * 32 * 4 * 2), d_power_size(7 * 32 * 4), d_time_scrunch_size(27 * 7), d_freq_scrunch_size(27 * 7), nstreams(4)
{
    cout << "Hello from the GPUpool constructor!!" << endl;
    cout.flush();
}

GPUpool::GPUpool(int ii) : d_fft_size(7 * 32 * 4 * 2), d_in_size(7 * 32 * 4 * 2), d_power_size(7 * 32 * 4), d_time_scrunch_size(27 * 7), d_freq_scrunch_size(27 * 7), nstreams(4), mainbuffer()
{


    cout << "Hello from the GPUpool constructor with id " << ii << "!!" << endl;
    cout.flush();
    id = ii;
    highest_frame = 0;
}

void GPUpool::execute(void)
{
    cout << "Setting device ID to " << id << endl;
    cudaSetDevice(id);

    std::shared_ptr<boost::asio::io_service> iop(new boost::asio::io_service);
    ios = iop;

    h_pol = new cufftComplex[d_in_size * 2];
    mystreams = new cudaStream_t[4];
    myplans = new cufftHandle[4];

    int nkernels = 3;
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];

    CUDAthreads[0] = 32 * 4 * 7;
    CUDAthreads[1] = 7;
    CUDAthreads[2] = 7 * (32 - 5);
    CUDAblocks[0] = 1;
    CUDAblocks[1] = 1;
    CUDAblocks[2] = 1;

    cudaHostAlloc((void**)&h_in, d_in_size * nstreams * sizeof(cufftComplex), cudaHostAllocDefault);
    cudaMalloc((void**)&d_in, d_in_size * nstreams * sizeof(cufftComplex));
    cudaMalloc((void**)&d_fft, d_fft_size * nstreams * sizeof(cufftComplex)); 
    
    dedisp_totsamples = (size_t)131072 + 5000;
    dedisp_buffno = (dedisp_totsamples - 1) / 131072 + 1;
    dedisp_buffsize = dedisp_buffno * 131072 + 5000;
    mainbuffer.allocate(dedisp_buffno, 4999, 131072, dedisp_buffsize, stokes);
   
    dv_power.resize(nstreams);
    dv_time_scrunch.resize(nstreams);
    dv_freq_scrunch.resize(nstreams);
    // TODO: make a private const data memmber and put in the initializer list!!
    stokes = 4;
    for (int ii = 0; ii < nstreams; ii++) {
        dv_power[ii].resize(d_power_size * stokes);
        dv_time_scrunch[ii].resize(d_time_scrunch_size * stokes);
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * stokes);
    }


    for (int ii = 0; ii < 4; ii++) {
        cudaStreamCreate(&mystreams[ii]);
        cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftpoint, NULL, 1, fftpoint, CUFFT_C2C, batchsize);
        cufftSetStream(myplans[ii], mystreams[ii]);
        mythreads.push_back(thread(&GPUpool::worker, this, ii));
    }

    boost::asio::socket_base::reuse_address option(true);
    boost::asio::socket_base::receive_buffer_size option2(9000);

    cout << "Creating sockets" << endl;
    cout.flush();

    for (int ii = 0; ii < 6; ii++) {
        sockets.push_back(udp::socket(*ios, udp::endpoint(boost::asio::ip::address::from_string("10.17.0.2"), 17100 + ii + 6 * id)));
        sockets[ii].set_option(option);
        sockets[ii].set_option(option2);
    }

    mythreads.push_back(thread(&GPUpool::receive_thread, this));
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ios->run();

    cout_guard.lock();
    cout << "Starting a GPU thread!!" << endl;
    cout_guard.unlock();
    buffer = new double[32];
    for (int ii = 0; ii < 32; ii++)
        buffer[ii] = id;
    std::this_thread::sleep_for(std::chrono::seconds(5));
}

void GPUpool::worker(int ii)
{
    cout << "Starting worker " << ii << " on GPU " << id << endl;
 
    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    float *pdv_power = thrust::raw_pointer_cast(dv_power[ii].data());
    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[ii].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[ii].data());

    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    unsigned int skip = ii * d_in_size;
    while(true) {
        unsigned int index{0};
        datamutex.lock();
        if(!mydata.empty()) {
            std::copy((mydata.front()).first.begin(), (mydata.front()).first.end(), h_in + skip);
            cout << "Copied the data" << endl;
            obs_time frame_time = mydata.front().second;
            mydata.pop();
            datamutex.unlock();
            cufftExecC2C(myplans[ii], d_in + skip, d_fft + skip, CUFFT_FORWARD);
            powerscale<<<CUDAblocks[0], CUDAthreads[0], 0, mystreams[ii]>>>(d_fft + skip, pdv_power, d_power_size);
            addtime<<<CUDAblocks[1], CUDAthreads[1], 0, mystreams[ii]>>>(pdv_power, pdv_time_scrunch, d_power_size, d_time_scrunch_size, 1);
            addchannel<<<CUDAblocks[2], CUDAthreads[2], 0, mystreams[ii]>>>(pdv_time_scrunch, pdv_freq_scrunch, d_time_scrunch_size, d_freq_scrunch_size, 1);
            cout << cudaGetErrorString(cudaGetLastError()) << endl;
            mainbuffer.write(pdv_freq_scrunch, frame_time, d_freq_scrunch_size, mystreams[ii]);
            cout << cudaGetErrorString(cudaGetLastError()) << endl;
            cudaDeviceSynchronize();
	    
	} else {
            datamutex.unlock();
            std::this_thread::yield();
        }
    }
}

GPUpool::~GPUpool(void)
{
    std::lock_guard<std::mutex> addguard(cout_guard);
    /*for (int ii = 0; ii < 32; ii++)
        cout << buffer[ii] << endl; */
    cout << "Bye fron the GPUpool destructor!!" << endl;
    for (int ii = 0; ii < mythreads.size(); ii++)
        mythreads[ii].join();
}

void GPUpool::receive_thread(void) {
    cout << "In the receiver thread. Waiting to get something..." << endl;
    cout.flush();
    sockets[5].async_receive_from(boost::asio::buffer(rec_buffer), sender_endpoint, boost::bind(&GPUpool::receive_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, sender_endpoint));
}

void GPUpool::receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint &endpoint) {
    header_s head;
    cout << "I'm in the handler" << endl;
    get_header(rec_buffer.data(), head);
    static obs_time start_time{head.epoch, head.ref_s};
    // this is ugly, but I don't have a better solution at the moment
    int long_ip = boost::asio::ip::address_v4::from_string((endpoint.address()).to_string()).to_ulong();
    int fpga = ((int)((long_ip >> 8) & 0xff) - 1) * 8 + ((int)(long_ip & 0xff) - 1) / 2;


    get_data(rec_buffer.data(), fpga, start_time);
    packetcount++;
    if (packetcount < 2)
        receive_thread();
}

void GPUpool::get_data(unsigned char* data, int fpga_id, obs_time start_time)
{
    // REMEMBER - d_in_size is the size of the single buffer (2 polarisations, 336 channels, 128 time samples)
    unsigned int idx = 0;
    unsigned int idx2 = 0;


    header_s head;
    get_header(data, head);

    // there are 250,000 frames per 27s period
    int frame = head.frame_no + (head.ref_s - start_time.start_second) * 250000;

    //int fpga_id = frame % 48;
    //int framet = (int)(frame / 48);         // proper frame number within the current period

    //int bufidx = frame % pack_per_buf;                                          // number of packet received in the current buffer

    //int fpga_id = thread / 7;       // - some factor, depending on which one is the lowest frequency

    //int fpga_id = frame % 48;
    //int framet = (int)(frame / 48);         // proper frame number within the current period

    int bufidx = fpga_id + (frame % 2) * 48;                                    // received packet number in the current buffer
    //int bufidx = frame % pack_per_buf;                                          // received packet number in the current buffer

    int startidx = ((int)(bufidx / 48) * 48 + bufidx) * WORDS_PER_PACKET;       // starting index for the packet in the buffer
                                                                                // used to skip second polarisation data

    // TEST: version for 7MHz band only
    if (frame > highest_frame) {

        highest_frame = frame;
        //highest_framet = (int)(frame / 48)

        #pragma unroll
        for (int chan = 0; chan < 7; chan++) {
            for (int sample = 0; sample < 128; sample++) {
                idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
                idx2 = chan * 128 + sample;        // get the position in the buffer
                h_pol[idx2].x = (float)(data[HEADER + idx + 0] | (data[HEADER + idx + 1] << 8));
                h_pol[idx2].y = (float)(data[HEADER + idx + 2] | (data[HEADER + idx + 3] << 8));
                h_pol[idx2 + d_in_size / 2].x = (float)(data[HEADER + idx + 4] | (data[HEADER + idx + 5] << 8));
                h_pol[idx2 + d_in_size / 2].y = (float)(data[HEADER + idx + 6] | (data[HEADER + idx + 7] << 8));
            }
        }

    }

    if ((frame % 2) == 0) {                     // send the first one
        add_data(h_pol, {start_time.start_epoch, start_time.start_second, frame});
        cout << "Sent the first buffer" << endl;
        cout.flush();
    } else if((frame % 2) == 1) {        // send the second one
        add_data(h_pol + d_in_size, {start_time.start_epoch, start_time.start_second, frame});
        cout << "Sent the second buffer" << endl;
        cout.flush();
    }
}

void GPUpool::add_data(cufftComplex *buffer, obs_time frame_time)
{
    std::lock_guard<mutex> addguard(datamutex);
    // TODO: is it possible to simplify this messy line?
    mydata.push(pair<vector<cufftComplex>, obs_time>(vector<cufftComplex>(buffer, buffer + d_in_size), frame_time));
}

int main(int argc, char *argv[])
{
    Oberpool();
    return 0;
}
