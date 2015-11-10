#include <cstdlib>
#include <cstring>  // memset
#include <iostream>
#include <string>


#include <arpa/inet.h>  // inet_ntop()
#include <errno.h>
#include <netdb.h>  // gethostbyname() etc..
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>   // waitpid()
#include <unistd.h> // close()
#include <signal.h>     // sigemptyset()

using std::cout;
using std::endl;

#define PORT "4000"   // just a random port number, must be passed as string to getaddrinfo
#define MAXBUFLEN 128

void *get_addr(sockaddr *sadr)
{
    if (sadr->sa_family == AF_INET) {
        return &(((sockaddr_in*)sadr)->sin_addr);
    }

    return &(((sockaddr_in6*)sadr)->sin6_addr);
}

struct mycomplex {

    float x;
    float y;

};

int main(int argc, char *argv[])
{

    const unsigned int fftsize = 8;
    const unsigned int batchsize = 1;    // the number of FFTs we want to perform at once
    const unsigned int timesamp = 1;		// the number fo timesamples we will store in buffer before processing
    const unsigned int totalsize = fftsize * batchsize * timesamp;
    const unsigned int memsize = totalsize * sizeof(mycomplex);

    mycomplex *floatbuf = new mycomplex[totalsize];

    int sfd, numbytes;
    socklen_t addrlen;     // socklen_t has length of at least 32 bits
    addrinfo hints, *servinfo, *p;
    sockaddr_storage their_addr;     // sockaddr_storage is large enough accommodate all supported  protocol-specific address structures
    char s[INET6_ADDRSTRLEN];   // length of the string form for IPv6
    char buf[MAXBUFLEN];


    memset(&hints, 0, sizeof(hints));   // will have problems without that
    hints.ai_family = AF_INET6;        // don't care if IPv4 or IPv6
    hints.ai_socktype = SOCK_DGRAM;     // datagram socket
    hints.ai_flags = AI_PASSIVE;        // can use NULL is getaddrinfo

    // ignore error checking for now
    getaddrinfo(NULL, PORT, &hints, &servinfo);

    // bind to the first possible result
    for(p = servinfo; p != NULL; p = p->ai_next) {

        if((sfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1 ) {
            cout << "error listener: socket" << endl;
            continue;
        }

        if (bind(sfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sfd);
            cout << "error listener: bind" << endl;
            continue;
        }

        break;

    }

    // failed miserably
    if (p == NULL) {
        cout << "listener: failed to bind socket" << endl;
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);     // no longer need this
    cout << "listener: waiting to recvfrom..." << endl;

    while(true)
    {
        if ((numbytes = recvfrom(sfd, floatbuf, memsize, 0, (struct sockaddr*)&their_addr, &addrlen)) == -1 ) {
            cout << "error recvfrom" << endl;
            exit(EXIT_FAILURE);
        }

        if (!numbytes)
            break;

        // convert IPv4 and IPv6 addresses between binary and test form
        // casting pointers is beautiful
        inet_ntop(their_addr.ss_family, get_addr((sockaddr*)&their_addr), s, sizeof(s));
        cout << "listener: got packet from " << s << endl;
        cout << "listener: packet is " << numbytes << " bytes long" << endl;
        cout << "listener: packet contains " << endl;
        for(int ii = 0; ii < totalsize; ii++) {
            cout << floatbuf[ii].x << " " << floatbuf[ii].y <<  endl;
            cout.flush();
        }
    }

    cout << "Stopped receiving stuff" << endl;

    delete [] floatbuf;

    close(sfd);

    return 0;
}
