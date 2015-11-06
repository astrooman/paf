#include <iostream>
#include <queue>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::thread;
using std::vector;

#define PORT "39478"

__global__ void poweradd(unsigned int jump);

class Pool
{
    private:
        bool working;
        // const to be safe
        const unsigned int batchsize;
        const unsigned int fftsize;
        const unsigned int timesamp;
        // one buffer
        unsigned int bufsize;
        unsigned int bufmem;
        // buffer for all streams together
        unsigned int totsize;
        unsigned int totmem;
        // GPU and thread stuff
        int sizes[1];
        int avt;
        cudaStream_t *mystreams;
        cufftHandle *myplans;
        mutex addmutex;
        mutex workmutex;
        unsigned int nthreads;
        unsigned int nblocks;
        // containers
        // use queue as FIFO needed
        queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
    protected:

    public:
        Pool(unsigned int bs, unsigned int fs, unsigned int ts);
        ~Pool(void);
        void add_data(cufftComplex *buffer);
        void minion(int stream);
};

Pool::Pool(unsigned int bs, unsigned int fs, unsigned int ts) : batchsize(bs),
                                                                fftsize(fs),
                                                                timesamp(ts),
                                                                working(true),
                                                                nthreads(256)
{
    cout << "Starting up. This may take few seconds..." << endl;
    avt = thread::hardware_concurrency();
    bufsize = fftsize * batchsize * timesamp;
    bufmem = bufsize * sizeof(cufftComplex);
    totsize = bufsize * avt;
    totmem = bufmem * avt;
    // / 2 as interested in time averaged output
    nblocks = (bufsize / 2 - 1 ) / nthreads + 1;

    sizes[0] = (int)fftsize;
    // want as many streams and plans as there will be threads
    // every thread will be associated with its own stream
    mystreams = new cudaStream_t[avt];
    myplans = new cufftHandle[avt];

    for (int ii = 0; ii < avt; ii++) {
        cudaStreamCreate(&mystreams[ii]);
        cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftsize, NULL, 1, fftsize, CUFFT_C2C, batchsize);
        cufftSetStream(myplans[ii], mystreams[ii]);
        // need to meet requirements for INVOKE(f, t1, t2, ... tn)
        // (t1.*f)(t2, ... tn) when f is a pointer to a member function of class T
        // and t1 is an object of type T or a reference to an object of type T
        // or a reference to an object of a type derived from T (C++14 §20.9.2)
        mythreads.push_back(thread(&Pool::minion, this, ii));

    }
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
    std::lock_guard<mutex> addguard(addmutex);
    // that has to have a mutex
    mydata.push(vector<cufftComplex>(buffer, buffer + bufsize));
}

void Pool::minion(int stream)
{
    cout << "Starting thread associated with stream " << stream << endl << endl;
    cout.flush();

    while(working) {
        // need to protect if with mutex
        if(!mydata.empty()) {
            //cudaMemcpyAsync(d_in, h_in, memsize, cudaMemcpyHostToDevice);
            poweradd<<<nblocks, nthreads, 0, mystreams[stream]>>>(fftsize * batchsize);
            //cudaMemcpyAsync(h_out, d_out, memsize / 2, cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();
        } else {
            std::this_thread::yield();
        }
    }
}

__global__ void poweradd(unsigned int jump)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int idx2 = blockIdx.x * blockDim.x + threadIdx.x + jump;
}

int main(int argc, char *argv[])
{
    // using thread pool will remove the need of checking which stream is used
    // each thread will be associated with a separate stream
    // it will start proceesing the new chunk as soon as possible
    Pool mypool(192, 32, 2);

    // networking stuff
    int sfd, numbytes, rv;
    socklen_t addrlen;     // socklen_t has length of at least 32 bits
    addrinfo hints, *servinfo, *p;
    sockaddr_storage their_addr;    // sockaddr_storage is large enough accommodate all supported
                                    //protocol-specific address structures
    char s[INET6_ADDRSTRLEN];   // length of the string form for IPv6

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET6;     //force IPv6 for now, as there were some problems with IPv4 on certain hardware
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;    // allows to use NULL in getaddrinfo

    if((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
        cout << "getaddrinfo error " << gai_strerror(rv) << endl;
        exit(EXIT_FAILURE);
    }

    // loop through the linked list and try binding to the first possible socket
    for (p = servinfo; p != NULL; p = p->ai_next) {
        if((sfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            perror("socket ");
            continue;
        }

        if(bind(sfd, pi->ai_addr, p->ai_addrlen) == -1) {
            close(sfd);
            perror("bind ");
            continue;
        }
        break;
    }

    if (p = NULL) {
        cout << "error: failed to bind the socket\n";
        exit(EXIT_FAILURE);
    }
    return 0;
}
