#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#define PORT "4000"     // random port number, same as listener's
#define BUFFER 16
using std::cout;
using std::endl;

struct mycomplex {

    float x;
    float y;

};

int main(int argc, char *argv[])
{

    const unsigned int fftsize = 4;
    const unsigned int batchsize = 1;    // the number of FFTs we want to perform at once
    const unsigned int timesamp = 1;		// the number fo timesamples we will store in buffer before processing
    const unsigned int totalsize = fftsize * batchsize * timesamp;
    const unsigned int memsize = totalsize * sizeof(mycomplex);
    int rv = 0;

    cout << "Will send " << memsize / 1024 << "K of data" << endl;

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 arreng{seed};
    std::normal_distribution<float> arrdis(0.0, 1.0);

    mycomplex *my_data = new mycomplex[totalsize];

    int sfd;
    addrinfo hints, *servinfo, *p;
    int numbytes;
    char buf[BUFFER];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET6;
    hints.ai_socktype = SOCK_DGRAM;

    // argv[1] is the IP of the lisener
    // no error checking for now
    getaddrinfo(argv[1], PORT, &hints, &servinfo);

    for (p = servinfo; p != NULL; p = p->ai_next) {

        if ((sfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1 ) {
            cout << "error talker: socket\n";
            continue;
        }

        break;

    }

    if (p == NULL) {
        cout << "error talker: failed to create socket\n";
        exit(EXIT_FAILURE);
    }

    std::stringstream ss;

    for (int packet = 0; packet < 2; packet++) {

        cout << "Will send:" << endl;
        for (int ii = 0; ii < totalsize; ii++) {
            my_data[ii].x = arrdis(arreng);
            my_data[ii].y = arrdis(arreng);
            cout << my_data[ii].x << " " << my_data[ii].y << endl;
        }
        ss.str("");

        ss << "packet " << packet;
        int size = ss.str().length();
        cout << size << endl;

        memcpy(buf, ss.str().c_str(), size);
        buf[size] = '\0';

        cout << buf << endl;

        if ((numbytes = sendto(sfd, (char*)my_data, memsize, 0, p->ai_addr, p->ai_addrlen)) == -1 ) {
            perror("talker: sento");
            cout << endl;
            exit(EXIT_FAILURE);
        }

        cout << "talker send " << numbytes << " bytes to " << argv[1] << endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // not overly good way of closing the connection
    if ((numbytes = sendto(sfd, buf, 0, 0, p->ai_addr, p->ai_addrlen)) == -1 ) {
        perror("talker: sendto");
        cout << endl;
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);

    cout << "talker send " << numbytes << " bytes to " << argv[1] << endl;

    delete [] my_data;

    return 0;
}
