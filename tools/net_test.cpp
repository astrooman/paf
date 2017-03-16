#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define PORTS 8
#define SIZE 32768

using std::cout;
using std::endl;
using std::mutex;
using std::thread;
using std::vector;

mutex printmutex;

void TestFpga(int ii, std::string strip) {

    printmutex.lock();
    cout << "Starting receive on port " << 17100 + ii << endl;
    printmutex.unlock();

    int sfd, netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    std::ostringstream oss;
    std::string strport;
    oss.str("");
    oss << 17100 + ii;
    strport = oss.str();


    if((netrv = getaddrinfo(strip.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
        cout << "getaddrinfo() error: " << gai_strerror(netrv) << endl;
    }

    for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
        if((sfd = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
            cout << "Socket error\n";
            continue;
        }

        if(bind(sfd, tryme->ai_addr, tryme->ai_addrlen) == -1) {
            close(sfd);
            cout << "Bind error\n";
            continue;
        }
        break;
    }

    if (tryme == NULL) {
        cout << "Failed to bind to the socket\n";
    }

    int fpga, numbytes;
    int *fpgaid = new int[6];
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    for (int ipack = 0; ipack < SIZE; ipack++) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        fpgaid[fpga % 6]++;
    }

    printmutex.lock();
    cout << "Received on port " << 17100 + ii << endl;
    for (int ifpga = 0; ifpga < 6; ifpga++) {
        cout << "FPGA " << ii * 6 + ifpga << ": " << fpgaid[ifpga] << endl;
    }
    cout << endl;
    printmutex.unlock();


}

void TestPort(int ii, std::string strip) {

    printmutex.lock();
    cout << "Starting receive on port " << 17100 + ii << endl;
    printmutex.unlock();

    int sfd, netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    std::ostringstream oss;
    std::string strport;
    oss.str("");
    oss << 17100 + ii;
    strport = oss.str();


    if((netrv = getaddrinfo(strip.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
        cout << "getaddrinfo() error: " << gai_strerror(netrv) << endl;
    }

    for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
        if((sfd = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
            cout << "Socket error\n";
            continue;
        }

        if(bind(sfd, tryme->ai_addr, tryme->ai_addrlen) == -1) {
            close(sfd);
            cout << "Bind error\n";
            continue;
        }
        break;
    }

    if (tryme == NULL) {
        cout << "Failed to bind to the socket\n";
    }

    int fpga, numbytes;
    size_t frameno = 0 , highestframe = 0;
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    for (int ipack = 0; ipack < 6 * SIZE; ipack++) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frameno = (size_t)(int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        if (frameno > highestframe)
            highestframe = frameno;
    }

    printmutex.lock();
    cout << (double)highestframe / (double)SIZE * 100.0 << "% received on port " << 17000 + ii << endl;
    cout << endl;
    printmutex.unlock();

}

int main(int argc, char *argv[])
{

    vector<thread> fpgathreads;
    vector<thread> portthreads;

    printmutex.lock();
    cout << "Recording data on " << argv[1] << endl;
    printmutex.unlock();


    // NOTE: this tests what percentage of data on a given port from each FPGA
    for (int iport = 0; iport < PORTS; iport++) {
        fpgathreads.push_back(thread(TestFpga, iport, std::string(argv[1])));
    }

    for (int iport = 0; iport < PORTS; iport++) {
        fpgathreads[iport].join();
    }

    // NOTE: this tests the percentage of data the port receives
    for (int iport = 0; iport < PORTS; iport++) {
        portthreads.push_back(thread(TestPort, iport, std::string(argv[1])));
    }

    for (int iport = 0; iport < PORTS; iport++) {
        portthreads[iport].join();
    }

    return 0;
}
