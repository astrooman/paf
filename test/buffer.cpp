#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <cufft.h>
#include <vdif_head.hpp>

#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define PORT 5000
#define BUF_LEN 8908 + 64   // 8908 bytes for sample block and 64  bytes for header
#define DATA 8908

using std::cout;
using std::endl;

void* get_addr(struct sockaddr *sa)
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
    sockaddr_storage their_arrd;
    socklen_t addr_len;
    char frame[BUF_LEN];
    char s[INET6_ADDRSTRLEN];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    getaddrinfo(NULL, PORT, &hints, &servinfo);

    for (p = servinfo; p != NULL; p = p->ai_next) {

        if((sfd = socket(p-ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
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
    int band;
    cufftComplex *data = new cufftComplex[896];
    while(true)
    {
        if ((numbytes = recvfrom(sfd, frame, MAXBUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addrlen)) == -1) {
            cout << "Error on recvfrom\n";
        }

        get_header(frame, head);
        get_data(frame, data)
        band = head.frame_no % 48;


        if (!numbytes)  // break on 0 bytes received for now - later process until the last frame reached
            break;

        inet_ntop(their_addr.ss_family, get_addr((sockaddr*)&their_addr), s, sizeof(s));

    }

    return 0;
}
