#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <cufft.h>
#include <pdif.hpp>

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

<<<<<<< HEAD
#define PORT "26666"
=======
#define PORT "17003"
>>>>>>> 210c63710f6857d9b1a88f552855e6b776ae05a9
#define DATA 8096
#define BUFLEN 7232   // 8908 bytes for sample block and 64  bytes for header

using std::cout;
using std::endl;

void* get_addr(struct sockaddr *sadr)
{
    if (sadr->sa_family == AF_INET) {
        return &(((sockaddr_in*)sadr)->sin_addr);
    }
    return &(((sockaddr_in6*)sadr)->sin6_addr);
}

int main(int argc, char *argv[])
{
    int sfd, numbytes, rv;
    addrinfo hints, *servinfo, *p;
    sockaddr_storage their_addr;
    socklen_t addr_len;
    char frame[BUFLEN];
    char s[INET6_ADDRSTRLEN];

    std::chrono::time_point<std::chrono::system_clock> recstart, recend;
    std::chrono::duration<double> recelapsed;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    if((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
	cout << "getaddrinfo error " << gai_strerror(rv) << endl;
	exit(EXIT_FAILURE);
    }

    for (p = servinfo; p != NULL; p = p->ai_next) {

        if((sfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            cout << "Socket error\n";
            continue;
        }

        if (bind(sfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sfd);
            cout << "Error on bind\n";
            continue;
        }

        break;
    }

    if (p == NULL) {
        cout << "Failed to bind the socket\n";
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);
    cout << "Waiting for recvfrom..." << endl;

    header_s head;
    int band{0};
    cufftComplex *poladata = new cufftComplex[896];
    cufftComplex *polbdata = new cufftComplex[896];

    cufftComplex *polafull = new cufftComplex[128 * 336];
    cufftComplex *polbfull = new cufftComplex[128 * 336];

    recstart = std::chrono::system_clock::now();

    int repeat = 5;

    int d_begin = 0;
    int previous_frame = -1;

    for (int times = 0; times < repeat; times++) {

	numbytes = 0;
	memset(&head, 0, sizeof(head));
	memset(frame, '\0', BUFLEN);
        if ((numbytes = recvfrom(sfd, frame, BUFLEN-1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout << "Error on recvfrom\n";
        }

	frame[BUFLEN] = '\0';
	cout << "Received " << numbytes << " bytes\n";
	cout << frame[0] << " " << frame[1] << " " << frame[2] << endl;
        if (!numbytes) { // break on 0 bytes received for now - later process until the last frame reached
            cout << "Not received anything\n";
            break;
        }

        //get_header(reinterpret_cast<unsigned char*>(frame), head);
	//cout << head.frame_no << endl;
	//cout << previous_frame << endl;
//        get_data(reinterpret_cast<unsigned char*>(frame), polafull, polbfull, d_begin, head.frame_no, previous_frame);

    }

    recend = std::chrono::system_clock::now();
    recelapsed = recend - recstart;

    cout << "Total time spent filling " << repeat << " buffers was " << recelapsed.count() << "s\n";
    cout << "It took " << recelapsed.count() / (double)repeat << "s to fill one buffer\n";

    close(sfd);

    return 0;
}
