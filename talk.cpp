#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#define PORT "4000"     // random port number, same as listener's
#define BUFFER 16
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    int sfd;
    addrinfo hints, *servinfo, *p;
    int numbytes;
    char buf[BUFFER];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
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

    for (int packet = 0; packet < 20; packet++) {

        ss.str("");

        ss << "packet " << packet;
        int size = ss.str().length();
        cout << size << endl;

        memcpy(buf, ss.str().c_str(), size);
        buf[size] = '\0';

        cout << buf << endl;

        if ((numbytes = sendto(sfd, buf, size, 0, p->ai_addr, p->ai_addrlen)) == -1 ) {
            cout << "error talker: sento\n";
            exit(EXIT_FAILURE);
        }

        cout << "talker send " << numbytes << " bytes to " << argv[1] << endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // not overly good way of closing the connection
    if ((numbytes = sendto(sfd, buf, 0, 0, p->ai_addr, p->ai_addrlen)) == -1 ) {
        cout << "error talker: sento\n";
        exit(EXIT_FAILURE);
    }

    freeaddrinfo(servinfo);

    cout << "talker send " << numbytes << " bytes to " << argv[1] << endl;

    return 0;
}
