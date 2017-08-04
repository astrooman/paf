#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
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

#include "myreceiver.hpp"

using std::condition_variable;
using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::vector;

condition_variable condvar;
mutex condmut;
mutex printmutex;

std::mutex mt;
std::condition_variable startcond;

bool ready = false;
bool working = true;
queue<short*> myqueue;

struct ObsTime {
    int refepoch;
    int refsecond;
    int frametime;
};

ObsTime obsstart;

MyReceiver::MyReceiver(std::string ipstr, std::vector<int> ports, std::chrono::system_clock::time_point start, int toread) {

}

MyReceiver::~MyReceiver(void) {

}

void MyReceiver::DoWork(int idx) {
    while(working) {
        std::unique_lock<mutex> condlock(condmut);
        condvar.wait(condlock, []{ return !myqueue.empty() || !working;});
        if (!myqueue.empty()) {
            short *localvar = myqueue.front();
            myqueue.pop();
            condlock.unlock();
            cout << "Thread " << idx << " can do work now" << endl;
            cout << "Thread " << idx << " received the following data: ";
            for (int idata = 0; idata < 4; idata++) {
                cout << *(localvar + idata) << " ";
            }
            cout << endl;
        }
    }
}

void MyReceiver::GetData(int iport) {

    printmutex.lock();
    cout << "Starting receive on port " << iport << endl;
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
    oss << iport;
    strport = oss.str();


    if((netrv = getaddrinfo(ipstr_.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
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
    int frameno = 0 , highestframe = 0;
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));
    addr_len = sizeof(their_addr);

    std::vector<std::pair<int, short>> framevals;

    std::this_thread::sleep_until(recordstart_);
    printmutex.lock();
    cout << "Started recording on port " << iport << endl;
    printmutex.unlock();

    if (iport == 17100) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::lock_guard<std::mutex> lg(mt);
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        obsstart.refepoch = (unsigned int)(rec_buf[12] >> 2);
        obsstart.refsecond = (unsigned int)(rec_buf[3] | (rec_buf[2] << 8) | (rec_buf[1] << 16) | ((rec_buf[0] & 0x3f) << 24));
        obsstart.frametime = (int)(int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        startcond.notify_all();
//        framevals.push_back(std::make_pair(0, fpga));
    } else {
        std::unique_lock<std::mutex> ul(mt);
        startcond.wait(ul, []{return obsstart.frametime != -1;});
    }

    unsigned int refsecond;

    for (int ipack = 0; ipack < 8 * toread_; ipack++) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frameno = (size_t)(int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        refsecond = (unsigned int)(rec_buf[3] | (rec_buf[2] << 8) | (rec_buf[1] << 16) | ((rec_buf[0] & 0x3f) << 24));
        frameno = frameno + (refsecond - obsstart.refsecond) / 27 * 250000 - obsstart.frametime;
        if (frameno >= 0)
            framevals.push_back(std::make_pair(frameno, fpga));
    }

    printmutex.lock();
    cout << "Done recording on port " << iport << endl;
    printmutex.unlock();

    oss.str("");
    oss << "frames_port_" << iport;

    std::string portfile;
    portfile = oss.str();
    std::ofstream outfile(portfile.c_str());

    for (auto iframe = framevals.begin(); iframe != framevals.end(); ++iframe)
        outfile << (*iframe).first << " " << (*iframe).second << std::endl;

    outfile.close();

}

int main(int argc, char *argv[])
{

    std::chrono::system_clock::time_point recordstart = std::chrono::system_clock::now();
    time_t t = std::chrono::system_clock::to_time_t(recordstart);
    cout << "It is currently: " << asctime(gmtime(&t)) << endl;


    int beam;
    int nofpgas;
    int noports;
    int toread;

    string inipstr;
    string portstr;

    vector<int> portvector;

    for (int iarg = 0; iarg < argc; iarg++) {
        if (string(argv[iarg]) == "-i") {
            iarg++;
            inipstr = string(argv[iarg]);
        } else if (string(argv[iarg]) == "-p") {
            iarg++;
            std::stringstream portss{string(argv[iarg])};
            while(std::getline(portss, portstr, ','))
                portvector.push_back(atoi(portstr.c_str()));
        } else if (string(argv[iarg]) == "-s") {
            iarg++;
            toread = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-r") {
            iarg++;
            string utcstr = string(argv[iarg]);
            std::tm utctm = {};
            strptime(utcstr.c_str(), "%Y-%0m-%0dT%0H:%0M:%0S", &utctm);
            recordstart = std::chrono::system_clock::from_time_t(mktime(&utctm));
        } else if (string(argv[iarg]) == "-h") {
            cout << "net_test [OPTIONS]\n"
                << "\t-i <ip>\n"
                << "\t-p <port1,port2,...,portn>\n"
                << "\t-s <packets to read>\n"
                << "\t-r <UTC time>\n";
            exit(EXIT_SUCCESS);
        }
    }

    printmutex.lock();
    cout << "Recording data from IP " << inipstr << endl;
    cout << "On ports: ";
    for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport)
        cout << *iport << ", ";
    cout << endl;
    printmutex.unlock();

    MyReceiver receive(inipstr, portvector, recordstart, toread);

    return 0;
}
