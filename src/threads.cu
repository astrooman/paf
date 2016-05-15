#include <cstdlib>
#include <iostream>
#include <queue>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "buffer.cuh"
#include "config.hpp"
//#include <dedisp/dedisp.hpp>
//#include <dedisp/DedispPlan.hpp>
#include "network.hpp"
#include "pdif.hpp"
#include "pool_multi.cuh"

// Heimdall headers - including might be a bit messy
#include <heimdall/params.hpp>
#include <heimdall/pipeline.hpp>

#include <boost/asio.hpp>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using boost::asio::ip::udp;
using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::vector;

#define PORT "17103"
#define DATA 7168          // 128 time samples, 7 channels per time sample, 64-bit words
#define BUFLEN 7168 + 64   // 8908 bytes for sample block and 64  bytes for header

void *get_addr(sockaddr *sadr)
{
    if (sadr->sa_family == AF_INET) {
        return &(((sockaddr_in*)sadr)->sin_addr);
    }

    return &(((sockaddr_in6*)sadr)->sin6_addr);
}

int main(int argc, char *argv[])
{
    std::string config_file;

    bool test{false};           // don't use test buffer by default
    bool verbose{false};        // don't use verbose mode by default
    unsigned int chunks{32};    // 32 chunks by default - this is just for testing purposes
    unsigned int streamno{4};   // 4 streams by default
    unsigned int beamno{3};     // 3 beams by default
    unsigned int times{4};      // 4 time samples by default
    unsigned int freq{16};       // no frequency averaging by default, at least 8, possibly 16
    // might be 336 / 168 or 384 / 192
    unsigned int nchans{336};   // number of 1MHz channels - might change
    unsigned int ngpus{3};      // number of GPUs

    // dedispersion parameters
    double band = 1.185;         // sampling rate for each band in MHz
    double dstart{0.0};
    double dend{4000.0};
    double foff{0.50};
    double ftop{1400.0};
    double tsamp = ((double)1.0 / (band * 1e+06) * (double)32.0);
    unsigned int filchans{nchans * 27 / freq};
    unsigned int gulp{131072};  // 2^17, equivalent to ~14s for 108us sampling time

    int *killmask = new int[filchans];

    config_s config;
    default_config(config);

    // too many parameters to load as arguments - use config file
    if (argc >= 2) {
        for (int ii = 0; ii < argc; ii++) {
            if (std::string(argv[ii]) == "--config") {      // configuration file
                ii++;
                config_file = std::string(argv[ii]);
                read_config(config_file, config);
                break;      // configuration file should have everything included
            }
            if (std::string(argv[ii]) == "-c") {      // the number of chunks to process
                ii++;
                config.chunks = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-s") {     // the number of streams to use
                ii++;
                config.streamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-b") {     // the number of beams to accept the data from
                ii++;
                config.beamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-t") {     // the number of time sample to average
                ii++;
                config.timesavg = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-f") {     // the number of frequency channels to average
                ii++;
                config.freqavg = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-n") {      // the number of GPUs to use
                ii++;
                config.ngpus = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-b") {     // use the test buffer
                config.test = true;
            } else if (std::string(argv[ii]) == "-v") {
                config.verbose = true;
            } else if (std::string(argv[ii]) == "-h") {
                cout << "Options:\n"
                        << "\t -v - use verbose mode\n"
                        << "\t -c - the number of chunks to process\n"
                        << "\t -b - the number of beams to process\n"
                        << "\t -t - the number of time samples to average\n"
                        << "\t -f - the number of frequency channels to average\n"
                        << "\t -n - the number of GPUs to use\n"
                        << "\t -s - the number of CUDA streams per GPU to use\n"
                        << "\t -b - use the test buffer\n"
                        << "\t --config - configuration file\n"
                        << "\t -h - print out this message\n\n";
                exit(EXIT_SUCCESS);
            }
        }

    }
    // should not take more than 5 seconds
    cout << "Starting up. This may take few seconds..." << endl;

    int devcount{0};
/*    cudaGetDeviceCount(&devcount);
    if (config.ngpus > devcount) {
        cout << "You can't use more GPUs than you have available!" << endl;
        config.ngpus = devcount;
    }
*/
    // using thread pool will remove the need of checking which stream is used
    // each thread will be associated with a separate stream
    // it will start proceesing the new chunk as soon as possible
    unsigned int batchs{config.beamno * config.nchans};      // # beams * 192 channels
                                            // need to decide how this data will be stored
    unsigned int ffts{32};
    Oberpool mypool(config);

    /*Pool mypool(batchs, ffts, config.times, config.freq, config.streamno, 4, config);

    boost::asio::io_service ios;
    Network incoming(ios);
    incoming.set_receive();

    // networking stuff
    // standard approach
    int sfd, numbytes, rv;
    socklen_t addrlen;              // socklen_t has length of at least 32 bits
    addrinfo hints, *servinfo, *p;
    sockaddr_storage their_addr;    // sockaddr_storage is large enough accommodate all supported
                                    //protocol-specific address structures
    char s[INET6_ADDRSTRLEN];       // length of the string form for IPv6
    unsigned char *inbuf = new unsigned char[BUFLEN];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
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

    cufftComplex *chunkbuf = new cufftComplex[batchs * ffts * times];



    //unsigned int mempacket = 6144;   // how many bytes per packet to read
    size_t memsize = batchs * ffts * times * sizeof(cufftComplex);
    //const unsigned int packets = memsize / mempacket;   // const to avoid accidental changes
                                                        // number of packets require to receive
                                                        // one data 'chunk', i.e. the amount of
                                                        // data required to performed filterbanking
                                                        // with averaging for all necessary beams and channels
    // unsigned int packetel = mempacket / sizeof(cufftComplex);

    int chunkno{0};

    header_s head;

    int polsize = nchans * times * 32;

    cufftComplex *pola = new cufftComplex[polsize];
    cufftComplex *polb = new cufftComplex[polsize];

    int highest_frame  = -1;
    int highest_framet = 0;

    int ihave = 0;

    // proper data receiving
    while(ihave < 48) {

        numbytes = recvfrom(sfd, inbuf, BUFLEN, 0, (struct sockaddr*)&their_addr, &addrlen);

        // assume last packet will have 0 bytes
        if(!numbytes)
            break;
        ihave++;
        get_header(inbuf, head);
        static obs_time start_time{head.epoch, head.ref_s};
        // adding the data is already included in the get_data() method
        // there are 250,000 frames for a period of 27 seconds
        mypool.get_data(inbuf, head.frame_no + (head.ref_s - start_time.start_second) * 250000, highest_frame, highest_framet, head.thread, start_time);
    }
    */
    // while(chunkno < chunks) {
    //     // will only receive 6 or 7 channels in one packet
    //     // will have to stitch them together
    //     for (unsigned int packetno  = 0; packetno < packets; packetno++) {
    //         if((numbytes = recvfrom(sfd, inbuf, mempacket, 0, (struct sockaddr*)&their_addr, &addrlen)) == -1 ) {
    //             cout << "error recvfrom" << endl;
    //             exit(EXIT_FAILURE);
    //         }
    //         // get the pdif header and strip it off the data
    //         get_header(inbuf, head);
    //         //cout << "Received packet " << packetno << " with " << numbytes << " bytes\n";
    //         //cout.flush();
    //         // I am not happy with the amount of copying done here and below
    //         // COMMENTED OUT FOR COMPILATION - READING PDIF FILES WILL BE SORTED OUT
    //         //std::copy(inbuf, inbuf + packetel, chunkbuf + packetno * packetel);
    //     }
    //
    //     mypool.add_data(chunkbuf);
    //     //cout << "Received chunk " << chunkno << endl;
    //     //cout.flush();
    //     chunkno++;
    //     // will send 0 bytes as a last packet to end the loop
    //     if(!numbytes)
    //         break;
    //     inet_ntop(their_addr.ss_family, get_addr((sockaddr*)&their_addr), s, sizeof(s));
    // }
    //
    // if(test) {
    //
    //     cout << "Test buffer\n";
    //     cout.flush();
    //     // sleep just in case processing is slow
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    //
    //     cufftComplex *testbuf = new cufftComplex[batchs * ffts * times * chunks];
    //
    //     unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    //     std::mt19937_64 bufeng{seed};
    //     std::normal_distribution<float> bufdis(0.0, 1.0);
    //
    //     cout << "Filling the test array...\n";
    //
    //     for (int ii = 0; ii < batchs * ffts * times * chunks; ii++) {
    //         testbuf[ii].x = bufdis(bufeng);
    //         testbuf[ii].y = bufdis(bufeng);
    //     }
    //
    //     chunkno = 0;
    //     while(chunkno < chunks) {
    //         mypool.add_data(testbuf + chunkno * batchs * ffts * times);
    //         chunkno++;
    //     }
    //
    // }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    cudaDeviceReset();

    return 0;
}
