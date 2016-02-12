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

Pool::Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int sn, unsigned int fr, config_s config) : batchsize(bs),
                                                                fftsize(fs),
                                                                timesamp(ts),
                                                                pol_begin(0),
                                                                working(true),
                                                                streamno(sn),
                                                                freqavg(fr),
                                                                nthreads(256),
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

    totsamples = (size_t)config.gulp + dedisp.get_max_delay();
    buffno = (totsamples - 1) / config.gulp + 1;
    size_t buffsize = buffno * config.gulp + dedisp.get_max_delay();
    mainbuffer.allocate(buffno, dedisp.get_max_delay(), config.gulp, buffsize);
    //if (false)       // switch off for now
    //    dedisp.set_killmask(killmask);
    // everything should be ready for dedispersion after this point

    set_search_params(&params, config);
    hd_create_pipeline(&pipeline, params)
    // everything should be ready for single pulse search after this point

    filsize = fftsize * batchsize * timesamp;
    bufmem = filsize * sizeof(cufftComplex);
    totsize = filsize * avt;
    totmem = bufmem * avt;
    // / 2 as interested in time averaged output
    nblocks = (filsize / 2 - 1 ) / nthreads + 1;

    sizes[0] = (int)fftsize;
    // want as many streams and plans as there will be threads
    // every thread will be associated with its own stream
    h_pol = new cufftComplex[filsize * 2];      // * 2 to deal with 2 polarisations
    mystreams = new cudaStream_t[avt];
    myplans = new cufftHandle[avt];

    cudaHostAlloc((void**)&h_in, totsize * sizeof(cufftComplex), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_out, totsize / 2 * sizeof(float), cudaHostAllocDefault);
    cudaMalloc((void**)&d_in, totsize * sizeof(cufftComplex));
    cudaMalloc((void**)&d_out, totsize * avt / 2 * sizeof(unsigned char));
    // change this later to deal with any input type;
    cudaMalloc((void**)&d_dedisp, totsamples * sizeof(unsigned char));
    cudaMalloc((void**)&d_search, config.gulp * dedisp.get_dm_count() * sizeof(unsigned char));
    // here only launch threads that will take care of filterbank
    for (int ii = 0; ii < avt - 2; ii++) {
        cudaStreamCreate(&mystreams[ii]);
        cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftsize, NULL, 1, fftsize, CUFFT_C2C, batchsize);
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
    mydata.push(vector<cufftComplex>(buffer, buffer + filsize));
}

void Pool::dedisp_thread(int dstream)
{
    int ready = mainbuffer.ready();
    if (ready) {
        mainbuffer.send(d_dedisp, ready, mystreams[dstream]);
        // TO DO: include data member with the number of gulps already dedispersed
        cout << "Dedispersing gulp " << endl;
        dedisp.execute(totsamples, d_dedisp, 8, d_search, 8, DEDISP_DEVICE_POINTERS);
    } else {
        std::this_thread::yield();
    }
}

void Pool::minion(int stream)
{
    cout << "Starting thread associated with stream " << stream << endl << endl;
    cout.flush();

    unsigned int skip = stream * filsize;
    unsigned int outmem = filsize / 2 * sizeof(float);

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
            if(cudaMemcpyAsync(d_in + skip, h_in + skip, bufmem, cudaMemcpyHostToDevice, mystreams[stream]) != cudaSuccess) {
		        cout << "HtD copy error on stream " << stream << " " << cudaGetErrorString(cudaGetLastError()) << endl;
		        cout.flush();
	        }
            if(cufftExecC2C(myplans[stream], d_in + skip, d_in + skip, CUFFT_FORWARD) != CUFFT_SUCCESS)
		          cout << "Error in FFT execution\n";
            poweradd<<<nblocks, nthreads, 0, mystreams[stream]>>>(d_in + skip, d_out + skip / 2, fftsize * batchsize);
            if(cudaMemcpyAsync(h_out + skip / 2, d_out + skip / 2, outmem, cudaMemcpyDeviceToHost, mystreams[stream]) != cudaSuccess) {
		        cout << "DtH copy error on stream " << stream << " " << cudaGetErrorString(cudaGetLastError()) << endl;
		        cout.flush();
	        }
            mainbuffer.write(d_out, index, filsize / 2, mystreams[stream]);
            cudaThreadSynchronize();
        } else {
	        datamutex.unlock();
            std::this_thread::yield();
        }
    }
}

void Pool::get_data(unsigned char* data, int frame)
{
    unsigned int idx = 0;
    unsigned int idx2 = 0;

    if((frame - previous_frame) > 1) {
        // count words only as one word provides one full time sample per polarisation
        pol_begin += (frame - previous_frame) * 7 * 128;
    } else {
        pol_begin += 7 * 128;
    }

    if(pol_bein >= filsize) {
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
            h_pol[idx2 + filsize].x = (float)(data[HEADER + idx + 4] | (data[HEADER + idx + 5] << 8));
            h_pol[idx2 + filsize].y = (float)(data[HEADER + idx + 6] | (data[HEADER + idx + 7] << 8));
        }
    }

    previous_frame = frame;
}

void Pool::search_thread(int sstream)
{
    // include check of some for here
    bool ready{true};
    if (ready) {
        // this need access to config - make config a data member
        cout << "Searching in the gulp " << endl;
        hd_execute(pipeline, d_dedisp, config.gulp, 8)
  } else {
        std::this_thread::yield();
  }
}
