#ifndef _H_PAFRB_POOL
#define _H_PAFRB_POOL

#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <buffer.hpp>
#include <config.hpp>
#include <cuda.h>
#include <cufft.h>
#include <DedispPlan.h>

using std::mutex;
using std::queue;
using std::thread;
using std::vector;

class Pool
{
    private:
        // that can be anything, depending on how many output bits we decide to use
        Buffer<unsigned char> mainbuffer;
        DedispPlan dedisp;
        hd_pipeline pipeline;
        hd_params params;

        bool working;
        // const to be safe
        const unsigned int batchsize;
        const unsigned int fftsize;
        const unsigned int timesamp;
        const unsigned int streamno;
        const unsigned int freqavg;
        // one buffer
        unsigned int filsize;
        unsigned int bufmem;
        // buffer for all streams together
        unsigned int totsize;
        unsigned int totmem;
        // GPU and thread stuff
        unsigned char *d_dedisp;
        unsigned char *d_search;
        cufftComplex *h_in, *d_in;
        unsigned char *h_out, *d_out;
        int sizes[1];
        int avt;
        cudaStream_t *mystreams;
        cufftHandle *myplans;
        mutex datamutex;
        mutex workmutex;
        unsigned int nthreads;
        unsigned int nblocks;
        // containers
        // use queue as FIFO needed
        queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
        unsigned int buffno;
        size_t totsamples;
    protected:

    public:
        Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int sn, unsigned int fr, config_s config);
        ~Pool(void);
        // add deleted copy, move, etc constructors
        void add_data(cufftComplex *buffer);
        void dedisp_thread(int dstream);
        void minion(int stream);
        void search_thread(int sstream);
};

#endif
