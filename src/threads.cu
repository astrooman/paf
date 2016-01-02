#include <cstdlib>
#include <iostream>
#include <queue>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>
#include <dedisp.h>
#include <DedispPlan.hpp>
#include <vdif_head.hpp>

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::vector;

#define PORT "45003"
#define SINGLE_GULP 131027      // number of time samples per single-pulse detection data chunk

__global__ void poweradd(cufftComplex *in, float *out, unsigned int jump);

void *get_addr(sockaddr *sadr)
{
    if (sadr->sa_family == AF_INET) {
        return &(((sockaddr_in*)sadr)->sin_addr);
    }

    return &(((sockaddr_in6*)sadr)->sin6_addr);
}
// make a template to support dfferent dedisp input types
template <class T>
class Buffer
{
    private:
        size_t start;
        size_t end;
        const size_t size;
        T *d_buf;
    protected:

    public:
        Buffer(size_t size);
        ~Buffer(void);

        void write(T *data);
        // add deleted copy, move, etc constructors
};

template<class T>
Buffer<T>::Buffer(size_t size) : size(size)
{
    start = 0;
    end = 0;
    cudaMalloc((void**)&d_buf, size * sizeof(T));
}

template<class T>
Buffer<T>::~Buffer()
{
    end = 0;
    cudaFree(d_buf);
}

template<class T>
void Buffer<T>::write(T *d_data, unsigned int amount)
{
    end = end + amount;
    cudaMemcpy
}

class Pool
{
    private:
        bool working;
        // const to be safe
        const unsigned int batchsize;
        const unsigned int fftsize;
        const unsigned int timesamp;
        const unsigned int streamno;
        const unsigned int freqavg;
        // one buffer
        unsigned int bufs;
        unsigned int bufsize;
        unsigned int bufmem;
        // buffer for all streams together
        unsigned int totsize;
        unsigned int totmem;
        // GPU and thread stuff
        cufftComplex *h_in, *d_in;
        float *h_out, *d_out;
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
    protected:

    public:
        Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int sn, unsigned int fr, unsigned int bn);
        ~Pool(void);
        // add deleted copy, move, etc constructors
        void add_data(cufftComplex *buffer);
        void minion(int stream);
};

Pool::Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int sn, unsigned int fr, unsigned int bn) : batchsize(bs),
                                                                fftsize(fs),
                                                                timesamp(ts),
                                                                working(true),
                                                                streamno(sn),
                                                                freqavg(fr),
                                                                bufs(bn),
                                                                nthreads(256)
{

    avt = min(streamno,thread::hardware_concurrency());
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

    cudaHostAlloc((void**)&h_in, totsize * sizeof(cufftComplex), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_out, totsize / 2 * sizeof(float), cudaHostAllocDefault);
    cudaMalloc((void**)&d_in, totsize * sizeof(cufftComplex));
    cudaMalloc((void**)&d_out, totsize * avt / 2 * sizeof(float));

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
    std::lock_guard<mutex> addguard(datamutex);
    // that has to have a mutex
    mydata.push(vector<cufftComplex>(buffer, buffer + bufsize));
    //cout << "Data added\n";
    //cout.flush();
}

void Pool::minion(int stream)
{
    cout << "Starting thread associated with stream " << stream << endl << endl;
    cout.flush();

    unsigned int skip = stream * bufsize;
    unsigned int outmem = bufsize / 2 * sizeof(float);

    while(working) {
        // need to protect if with mutex
        // current mutex implementation is a big ugly, but just need a dirty hack
        // will write a new, thread-safe queue implementation
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
            cudaThreadSynchronize();
        } else {
	        datamutex.unlock();
            std::this_thread::yield();
        }
    }
}

__global__ void poweradd(cufftComplex *in, float *out, unsigned int jump)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int idx2 = blockIdx.x * blockDim.x + threadIdx.x + jump;

    if (idx1 < jump) {      // half of the input data
        float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
        float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
        out[idx1] = (power1 + power2) / 2.0;
    }
}

int main(int argc, char *argv[])
{
    std::string config_file;

    bool test{false};           // don't use test buffer by default
    bool verbose{false};        // don't use verbose mode by default
    unsigned int chunks{32};    // 32 chunks by default - this is just for testing purposes
    unsigned int streamno{4};   // 4 streams by default
    unsigned int beamno{3};     // 3 beams by default
    unsigned int times{4};      // 2 time samples by default
    unsigned int freq{8};       // no frequency averaging by default, at least 8, possibly 16
    // might be 336 / 168 or 384 / 192
    unsigned int nchans{192};   // number of 1MHz channels - might change

    // dedispersion parameters
    double band = 1.185;         // sampling rate for each band in MHz
    double dstart{0.0};
    double dend{4000.0};
    double foff{0.0};
    double ftop{0.0};
    double tsamp = ((double)1.0 / (band * 1e+06) * (double)32.0);
    unsigned int filchans{nchans * 27 / freq};
    unsigned int gulp{131072};  // 2^17, equivalent to ~14s for 108us sampling time

    int *killmask = new int[filchans];

    // too many parameters to load as arguments - use config file
    if (argc >= 2) {
        for (int ii = 0; ii < argc; ii++) {
            if (std::string(argv[ii]) == "--config") {      // configuration file
                ii++;
                config_file = std::string(argv[ii]);
                break;      // configuration file should have everything included
            }
            if (std::string(argv[ii]) == "-c") {      // the number of chunks to process
                ii++;
                chunks = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-s") {     // the number of streams to use
                ii++;
                streamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-b") {     // the number of beams to accept the data from
                ii++;
                beamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-t") {     // the number of time sample to average
                ii++;
                times = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-f") {     // the number of frequency channels to average
                ii++;
                freq = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-b") {     // use the test buffer
                test = true;
            } else if (std::string(argv[ii]) == "-v") {
                verbose = true;
            } else if (std::string(argv[ii]) == "-h") {
                cout << "Options:\n"
                        << "\t -v - use verbose mode\n"
                        << "\t -c - the number of chunks to process\n"
                        << "\t -b - the number of beams to process\n"
                        << "\t -t - the number of time samples to average\n"
                        << "\t -f - the number of frequency channels to average\n"
                        << "\t -s - the number of CUDA streams to use\n"
                        << "\t -b - use the test buffer\n"
                        << "\t -h - print out this message\n"
                        << "\t --config - configuration file\n\n";
                exit(EXIT_SUCCESS);
            }
        }

    }
    // should not take more than 5 seconds
    cout << "Starting up. This may take few seconds..." << endl;
    // tsamp in seconds, ftop and foff in MHz
    DedispPlan dedisp(filchans, tsamp, ftop, foff);
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials
    dedisp.generate_dm_list(dstart, dend, (float)64.0, (float)1.10);
    size_t buffsize = (size_t)gulp + dedisp.get_max_delay();
    unsigned int buffno = (buffsize - 1) / gulp + 1;
    cout << "Will try " << dedisp.get_dm_count() << " DM trials" << endl;
    if (verbose) {
        cout << "Will try " << dedisp.get_dm_count() << " DM trials:\n";
        for (int ii = 0; ii < dedisp.get_dm_count(); ii++)
            cout << *(dedisp.get_dm_list() + ii) << endl;
    }
    if (false)       // switch off for now
        dedisp.set_killmask(killmask);
    // everything should be ready for dedispersion after this point

    // using thread pool will remove the need of checking which stream is used
    // each thread will be associated with a separate stream
    // it will start proceesing the new chunk as soon as possible
    unsigned int batchs{beamno * nchans};      // # beams * 192 channels
                                            // need to decide how this data will be stored
    unsigned int ffts{32};
    Pool mypool(batchs, ffts, times, streamno, freq, buffno);

    // networking stuff
    int sfd, numbytes, rv;
    socklen_t addrlen;              // socklen_t has length of at least 32 bits
    addrinfo hints, *servinfo, *p;
    sockaddr_storage their_addr;    // sockaddr_storage is large enough accommodate all supported
                                    //protocol-specific address structures
    char s[INET6_ADDRSTRLEN];       // length of the string form for IPv6
    cufftComplex *chunkbuf = new cufftComplex[batchs * ffts * times];
    unsigned int mempacket = 6144;   // how many bytes per packet to read
    size_t memsize = batchs * ffts * times * sizeof(cufftComplex);
    const unsigned int packets = memsize / mempacket;   // const to avoid accidental changes
                                                        // number of packets require to receive
                                                        // one data 'chunk', i.e. the amount of
                                                        // data required to performed filterbanking
                                                        // with averaging for all necessary beams and channels
    unsigned int packetel = mempacket / sizeof(cufftComplex);
    cufftComplex *inbuf = new cufftComplex[packetel];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;    // allows to use NULL in getaddrinfo

    cout << "Will process " << memsize / 1024.0 << "KB chunks\n"
		<< "Divided into " << packets << " 6144B packets\n";

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

        if(bind(sfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sfd);
            perror("bind ");
            continue;
        }
        break;
    }
    // didn't bind to anything
    if (p == NULL) {
        cout << "error: failed to bind the socket\n";
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);     // no longer need it
    cout << "Waiting to receive from the server...\n";

    int chunkno{0};

    header_s head;

    while(chunkno < chunks) {

        for (unsigned int packetno  = 0; packetno < packets; packetno++) {
            if((numbytes = recvfrom(sfd, inbuf, mempacket, 0, (struct sockaddr*)&their_addr, &addrlen)) == -1 ) {
                cout << "error recvfrom" << endl;
                exit(EXIT_FAILURE);
            }
            // get the vdif header and strip it off the data
            get_header(inbuf, head);
            //cout << "Received packet " << packetno << " with " << numbytes << " bytes\n";
            //cout.flush();
            // I am not happy with the amount of copying done here and below
            std::copy(inbuf, inbuf + packetel, chunkbuf + packetno * packetel);
        }

        mypool.add_data(chunkbuf);
        //cout << "Received chunk " << chunkno << endl;
        //cout.flush();
        chunkno++;
        // will send 0 bytes as a last packet to end the loop
        if(!numbytes)
            break;
        inet_ntop(their_addr.ss_family, get_addr((sockaddr*)&their_addr), s, sizeof(s));
    }

    if(test) {

        cout << "Test buffer\n";
        cout.flush();
        // sleep just in case processing is slow
        std::this_thread::sleep_for(std::chrono::seconds(1));

        cufftComplex *testbuf = new cufftComplex[batchs * ffts * times * chunks];

        unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 bufeng{seed};
        std::normal_distribution<float> bufdis(0.0, 1.0);

        cout << "Filling the test array...\n";

        for (int ii = 0; ii < batchs * ffts * times * chunks; ii++) {
            testbuf[ii].x = bufdis(bufeng);
            testbuf[ii].y = bufdis(bufeng);
        }

        chunkno = 0;
        while(chunkno < chunks) {
            mypool.add_data(testbuf + chunkno * batchs * ffts * times);
            chunkno++;
        }

    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    cudaDeviceReset();

    return 0;
}
