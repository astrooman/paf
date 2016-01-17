#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <cufft.h>
#include <vdif.hpp>

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define PORT "5000"
#define DATA 8908
#define BUFLEN 8908 + 64   // 8908 bytes for sample block and 64  bytes for header

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
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    getaddrinfo(NULL, PORT, &hints, &servinfo);

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
        cout << "Faile to bind the socket\n";
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

    for (int times = 0; times < 20; times++) {

        while(band < 48) {
            if ((numbytes = recvfrom(sfd, frame, BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
                cout << "Error on recvfrom\n";
            }

            if (!numbytes)  // break on 0 bytes received for now - later process until the last frame reached
                break;

            get_header(reinterpret_cast<unsigned char*>(frame), head);
            get_data(reinterpret_cast<unsigned char*>(frame), poladata, polbdata);
            //band = head.frame_no % 48;
            std::copy(poladata, poladata + 896, polafull + band * 896);
            std::copy(polbdata, polbdata + 896, polbfull + band * 896);
            band++;

            inet_ntop(their_addr.ss_family, get_addr((sockaddr*)&their_addr), s, sizeof(s));

        }

    }

    recend = std::chrono::system_clock::now();
    recelapsed = recend - recstart;

    cout << "It took " << recelapsed.count() / (double)20.0 << "s to fill one buffer\n";

    return 0;
}
