#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <buffer.hpp>
#include <config.hpp>
#include <cuda.h>
#include <cufft.h>
#include <dedisp.h>
#include <DedispPlan.h>
#include <pool.hpp>

using std::mutex;
using std::queue;
using std::thread;
using std::vector;

#define BYTES_PER_WORD 8
#define WORDS_PER_PACKET 896

Pool::Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int fr, unsigned int sn, unsigned int np, config_s config) : batchsize(bs),
                                                                fftpoint(fs),
                                                                timeavg(ts),
                                                                freqavg(fr),
                                                                nostreams(sn),
                                                                npol(np),
                                                                d_in_size(bs * fs * ts * np),
                                                                d_fft_size(bs * fs * ts * np),
                                                                d_power_size(bs * fs * ts),
                                                                d_time_scrunch_size((bs - 5) * bs),
                                                                d_freq_scrunch_size((bs - 5) * bs / fr),
                                                                pol_begin(0),
                                                                gulps_processed(0),
                                                                working(true),
                                                                mainbuffer(),
                                                                dedisp(config.filchans, config.tsamp, config.ftop, config.foff)
{

    // streamno for filterbank and additional 2 for dedispersion and single pulse search
    avt = min(streamno + 2,thread::hardware_concurrency());

    if(config.verbose)
        cout << "Will create " << avt << " CUDA streams\n";

    // gemerate_dm_list(dm_start, dm_end, width, tol)
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials
    dedisp.generate_dm_list(config.dstart, config.dend, (float)64.0, (float)1.10);
    if (config.verbose) {
        cout << "Will try " << dedisp.get_dm_count() << " DM trials:\n";
        for (int ii = 0; ii < dedisp.get_dm_count(); ii++)
            cout << *(dedisp.get_dm_list() + ii) << endl;
    }

    dedisp_totsamples = (size_t)config.gulp + dedisp.get_max_delay();
    dedisp_buffno = (dedisp_totsamples - 1) / config.gulp + 1;
    dedisp_buffsize = dedisp_buffno * config.gulp + dedisp.get_max_delay();
    int stokes = 4; // make it a parameter later
    mainbuffer.allocate(dedisp_buffno, dedisp.get_max_delay(), config.gulp, dedisp_buffsize, stokes);
    //if (false)       // switch off for now
    //    dedisp.set_killmask(killmask);
    // everything should be ready for dedispersion after this point

    set_search_params(&params, config);
    hd_create_pipeline(&pipeline, params)
    // everything should be ready for single pulse search after this point

    sizes[0] = (int)fftpoint;
    // want as many streams and plans as there will be threads
    // every thread will be associated with its own stream
    h_pol = new cufftComplex[d_in_size];
    mystreams = new cudaStream_t[avt];
    myplans = new cufftHandle[avt];

    int nkernels = 3;
    nthreads = new unsigned int[nkernels];
    // nthreads[0] - powerscale() kernel; nthreads[1] - addtime() kernel; nthreads[2] - addchannel() kernel
    nblocks = new unsigned int[nkernels];
    // nblocks[0] - powerscale() kernel; nblocks[1] - addtime() kernel; nblocks[2] - addchannel() kernel
    // very simple approach for now
    nthreads[0] = fftpoint * timeavg * nchans;
    nthreads[1] = nchans;
    nthreads[2] = nchans * (fftpoint - 5) / freqavg;
    nblocks[0] = 1;
    nblocks[1] = 1;
    nblocks[2] = 1;

//  This is so broken, I am surprised at my own stupidity
/*    dv_power.resize(stokes);
    for (int ii = 0; ii < stokes; ii++)
        dv_power[ii].resize(d_power_size * streamno);
    pdv_power = thrust::raw_pointer_cast(dv_power.data());

    dv_time_scrunch.resize(stokes);
    for (int ii = 0; ii < stokes; ii++)
        dv_time_scrunch[ii].resize(d_time_scrunch_size * streamno);
    pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch);

    dv_freq_scrunch.resize(stokes);
    for (int ii = 0; ii < stokes)
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * streamno);
    pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch);
*/

    dv_power.resize(streamno);
    dv_time_scrunch.resize(streamno);
    dv_freq_scrunch.resize(streamno);
    for (int ii = 0; ii < streamno; ii++) {
        dv_power[ii].resize(d_power_size * stokes);
        dv_time_scrunch[ii].resize(d_time_scrunch_size * stokes);
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * stokes);
    }



    cudaHostAlloc((void**)&h_in, d_in_size * streamno * sizeof(cufftComplex), cudaHostAllocDefault);
    cudaMalloc((void**)&d_in, d_in_size * streamno * sizeof(cufftComplex));
    cudaMalloc((void**)&d_fft, d_fft_size * streamno * sizeof(cufftComplex));
    cudaMalloc((void**)&d_power, d_power_size * streamno * sizeof(float));
    cudaMalloc((void**)&d_time_scrunch, d_time_scrunch_size * streamno * sizeof(float));
    cudaMalloc((void**)&d_freq_scrunch, d_freq_scrunch_size * streamno * sizeof(float));
    // change this later to deal with any input type;
    cudaMalloc((void**)&d_dedisp, dedisp_totsamples * sizeof(unsigned char));
    cudaMalloc((void**)&d_search, config.gulp * dedisp.get_dm_count() * sizeof(unsigned char));
    // here only launch threads that will take care of filterbank
    for (int ii = 0; ii < avt - 2; ii++) {
        cudaStreamCreate(&mystreams[ii]);
        cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftpoint, NULL, 1, fftpoint, CUFFT_C2C, batchsize);
        cufftSetStream(myplans[ii], mystreams[ii]);
        // need to meet requirements for INVOKE(f, t1, t2, ... tn)
        // (t1.*f)(t2, ... tn) when f is a pointer to a member function of class T
        // and t1 is an object of type T or a reference to an object of type T
        // this is t1 and &Pool::minion is a pointer to a member function of class T
        // or a reference to an object of a type derived from T (C++14 §20.9.2)
        mythreads.push_back(thread(&Pool::minion, this, ii));
    }
    // dedisp thread
    cudaStreamCreate(&mystreams[avt-2]);
    mythreads.push_back(thread(&Pool::dedisp_thread, this, avt-2));
    // single pulse thread
    cudaStreamCreate(&mystreams[avt-1]);
    mythreads.push_back(thread(&Pool::search_thread, this, avt-1));
}

Pool::~Pool(void)
{
    working = false;
    // join the threads so main() thread will wait until all 4 last pieces are processed
    for (int ii = 0; ii < avt; ii++)
        mythreads[ii].join();
}

void Pool::add_data(cufftComplex *buffer)
{
    std::lock_guard<mutex> addguard(datamutex);
    // that has to have a mutex
    mydata.push(vector<cufftComplex>(buffer, buffer + d_in_size));
}



void Pool::minion(int stream)
{
    cout << "Starting thread associated with stream " << stream << endl << endl;
    cout.flush();

    // skip for data with two polarisations
    unsigned int skip = stream * d_in_size;

    float *pdv_power = thrust::raw_pointer_cast(dv_power[stream].data());
    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

    while(working) {
        // need to protect if with mutex
        // current mutex implementation is a big ugly, but just need a dirty hack
        // will write a new, thread-safe queue implementation
        unsigned int index{0};       // index will be used to distinguish between time samples
        datamutex.lock();
        if(!mydata.empty()) {
            std::copy((mydata.front()).begin(), (mydata.front()).end(), h_in + skip);
            mydata.pop();
            datamutex.unlock();
            //cout << "Stream " << stream << " got the data\n";
	        //cout.flush();
            if(cudaMemcpyAsync(d_in + skip, h_in + skip, d_in_size * sizeof(cufftComplex), cudaMemcpyHostToDevice, mystreams[stream]) != cudaSuccess) {
		        cout << "HtD copy error on stream " << stream << " " << cudaGetErrorString(cudaGetLastError()) << endl;
		        cout.flush();
	        }
            if(cufftExecC2C(myplans[stream], d_in + skip, d_fft + skip, CUFFT_FORWARD) != CUFFT_SUCCESS)
		          cout << "Error in FFT execution\n";
            powerscale<<<nblocks[0], nthreads[0], 0, mystreams[stream]>>>(d_fft + skip, pdv_power, d_power_size);
            //powerscale<<<nblocks[0], nthreads[0], 0, mystreams[stream]>>>(d_fft + skip, d_power + skip / npol, d_in_size / npol);
            addtime<<<nblocks[1], nthreads[1], 0, mystreams[stream]>>>(pdv_power, pdv_time_scrunch, d_power_size, d_time_scrunch_size, timeavg);
            //addtime<<<nblocks[1], nthreads[1], 0, mystreams[stream]>>>(d_power + skip / npol, d_time_scrunch + stream * d_time_scrunch_size);
            addchannel<<<nblocks[2]. nthreads[2], 0, mystreams[stream]>>>(pdv_time_scrunch, pdv_freq_scrunch, d_time_scrunch_size, d_freq_scrunch_size, freqavg)
            //addchannel<<<nblocks[2], nthreads[2], 0, mystreams[stream]>>>(d_time_scrunch + stream * d_time_scrunch_size, d_freq_scrunch + stream * );
            mainbuffer.write(pdv_freq_scrunch, unsigned int index, d_freq_scrunch_size, mystreams[stream]);
            cudaThreadSynchronize();
        } else {
	        datamutex.unlock();
            std::this_thread::yield();
        }
    }
}

void Pool::dedisp_thread(int dstream)
{
    while(working) {
        int ready = mainbuffer.ready();
        if (ready) {
            mainbuffer.send(d_dedisp, ready, mystreams[dstream]);
            // TO DO: include data member with the number of gulps already dedispersed
            cout << "Dedispersing gulp " << endl;
            dedisp.execute(dedisp_totsamples, d_dedisp, 8, d_search, 8, DEDISP_DEVICE_POINTERS);
        } else {
            std::this_thread::yield();
        }
    }
}

void Pool::search_thread(int sstream)
{
    while(working) {
        bool ready{true};
        if (ready) {
            // this need access to config - make config a data member
            cout << "Searching in the gulp " << endl;
            hd_execute(pipeline, d_dedisp, config.gulp, 8, gulps_processed);
            gulps_processed++;
        } else {
            std::this_thread::yield();
        }
  }
}

void Pool::get_data(unsigned char* data, int frame, int &previous_frame, obs_time start_time)
{
    unsigned int idx = 0;
    unsigned int idx2 = 0;

    if((frame - previous_frame) > 1) {
        // count words only as one word provides one full time sample per polarisation
        pol_begin += (frame - previous_frame) * 7 * 128;
    } else {
        pol_begin += 7 * 128;
    }

    // send the data to the data queue
    if(pol_bein >= d_in_size / 2) {
        add_data(h_pol);
        pol_begin = 0;
    }

    int fpga_id = frame % 48;
    #pragma unroll
    for (int chan = 0; chan < 7; chan++) {
        for (int sample = 0; sample < 128; sample++) {
            idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
            idx2 = chan * 128 + sample + fpga_id * WORDS_PER_PACKET;        // get the position in the buffer
            h_pol[idx2].x = (float)(data[HEADER + idx + 0] | (data[HEADER + idx + 1] << 8));
            h_pol[idx2].y = (float)(data[HEADER + idx + 2] | (data[HEADER + idx + 3] << 8));
            h_pol[idx2 + d_in_size / 2].x = (float)(data[HEADER + idx + 4] | (data[HEADER + idx + 5] << 8));
            h_pol[idx2 + d_in_size / 2].y = (float)(data[HEADER + idx + 6] | (data[HEADER + idx + 7] << 8));
        }
    }

    previous_frame = frame;
}
