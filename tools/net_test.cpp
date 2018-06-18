#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>
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

using std::cerr;
using std::cout;
using std::endl;
using std::mutex;
using std::string;
using std::thread;
using std::vector;

mutex printmutex;

#define SIZE 16384

std::mutex mt;
std::condition_variable startcond;

struct ObsTime {
    int refepoch;
    int refsecond;
    int frametime;
};

ObsTime obsstart;

void TestFpga(int iport, int usefpga, std::string strip, int toread) {

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

    int numbytes;
    int *fpgaid = new int[6];
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    //unsigned char *inbuffer = new unsigned char[(7168 + 64) * nofpgas * toread];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    size_t frameno = 0, highestframe = 0;
    unsigned int timestamp{0};

    unsigned short beamno{0};
    unsigned short fpga{0};
    unsigned short refepoch{0};

    vector<std::tuple<unsigned short, unsigned short, unsigned short, unsigned int, unsigned int>> headvals;

    int ipack = 0;

    while (true) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        frameno = (unsigned int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        if (frameno == 0) {
            cout << "Reached the 27s boundary. Will start recording now...\n";
            break;
        }
    }

    while (ipack < toread ) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((unsigned short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        beamno = (unsigned short)(rec_buf[23] | (rec_buf[22] << 8));
        if (fpga ==  usefpga)  {
            refepoch = (unsigned short)(rec_buf[12] >> 2);
            timestamp = (unsigned int)(rec_buf[3] | (rec_buf[2] << 8) | (rec_buf[1] << 16) | ((rec_buf[0] & 0x3f) << 24));
            frameno = (size_t)(int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
            if (frameno > highestframe) {
                highestframe = frameno;
            }
            //headvals.push_back(std::make_tuple(fpga, beamno, refepoch, timestamp, framestamp));
            ipack++;
        }
    }

    printmutex.lock();
    cout << "Received " << (float)toread / (float)(highestframe + 1) * 100.0f << "% on FPGA " << usefpga << endl;
    printmutex.unlock();

    // std::ofstream outhead("headers.codif", std::ios_base::out | std::ios_base::trunc);
    // if (outhead) {
    //     for (auto ihead = headvals.begin(); ihead != headvals.end(); ++ihead) {
    //         outhead << std::get<0>(*ihead) << " " << std::get<1>(*ihead) << " "
    //                     << std::get<2>(*ihead) << " " << std::get<3>(*ihead) << " "
    //                     << std::get<4>(*ihead) << endl;
    //     }
    // }
    // outhead.close();

/*    std::ofstream outdata("data.codif", std::ios_base::out | std::ios_base::binary);
    if (outdata) {
        outdata.write(reinterpret_cast<char*>(inbuffer), (7168 + 64) * nofpgas * toread);
    }
    outdata.close();
*/
    //delete [] inbuffer;
    delete [] rec_buf;

}

void TestPort(int iport, std::string strip, unsigned int toread, std::chrono::system_clock::time_point recordstart) {

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
    int frameno = 0 , highestframe = 0;
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));
    addr_len = sizeof(their_addr);

    std::vector<std::pair<int, short>> framevals;

    std::this_thread::sleep_until(recordstart);
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

    for (int ipack = 0; ipack < 8 * toread; ipack++) {
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

void TestBand(int iport, std::string strip, int usebeam, int toread, unsigned int *topframes) {

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

    int numbytes;
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    unsigned int framestamp{0};
    unsigned int timestamp{0};

    unsigned short beamno{0};
    unsigned short fpga{0};
    unsigned short refepoch{0};

    int ipack = 0;
    unsigned int starttime;

    while (true) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        framestamp = (unsigned int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        timestamp = (unsigned int)(rec_buf[3] | (rec_buf[2] << 8) | (rec_buf[1] << 16) | ((rec_buf[0] & 0x3f) << 24));
        if (framestamp == 0) {
            cout << "Reached the 27s boundary. Will start recording now...\n";
            starttime = timestamp;
            break;
        }
    }

    while (ipack < toread ) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((unsigned short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        beamno = (unsigned short)(rec_buf[23] | (rec_buf[22] << 8));
        if (beamno == usebeam) {
            refepoch = (unsigned short)(rec_buf[12] >> 2);
            timestamp = (unsigned int)(rec_buf[3] | (rec_buf[2] << 8) | (rec_buf[1] << 16) | ((rec_buf[0] & 0x3f) << 24));
            framestamp = (unsigned int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
            if (fpga == 8)
                fpga--;
            topframes[fpga] = framestamp + (timestamp - starttime) / 27 * 250000;
            ipack++;
        }
    }

    delete [] rec_buf;

}

int main(int argc, char *argv[])
{

    int beam;
    int nofpgas;
    int noports;
    int toread;

    memset(&obsstart, sizeof(obsstart), 0);
    obsstart.frametime = -1;

    std::chrono::system_clock::time_point recordstart = std::chrono::system_clock::now();
    time_t t = std::chrono::system_clock::to_time_t(recordstart);
    cout << "It is currently: " << asctime(gmtime(&t)) << endl;

    string fpgastr;
    string inipstr;
    string portstr;
    string testtype;

    vector<int> fpgavector;
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
        } else if (string(argv[iarg]) == "-f") {
            iarg++;
            std::stringstream fpgass{string(argv[iarg])};
            while(std::getline(fpgass, fpgastr, ','))
                fpgavector.push_back(atoi(fpgastr.c_str()));
        } else if (string(argv[iarg]) == "-s") {
            iarg++;
            toread = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-b") {
            iarg++;
            beam = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-r") {
            iarg++;
            string utcstr = string(argv[iarg]);
            std::tm utctm = {};
            strptime(utcstr.c_str(), "%Y-%0m-%0dT%0H:%0M:%0S", &utctm);
            recordstart = std::chrono::system_clock::from_time_t(mktime(&utctm));
        } else if (string(argv[iarg]) == "-t") {
            iarg++;
            testtype = string(argv[iarg]);
        } else if (string(argv[iarg]) == "-h") {
            cout << "net_test [OPTIONS]\n"
                << "\t-i <ip>\n"
                << "\t-p <port1,port2,...,portn>\n"
                << "\t-s <packets to read>\n"
                << "\t-b <beam to use>\n"
                << "\t-r <UTC time>\n"
                << "\t-t <p | f>\n\n";
            exit(EXIT_SUCCESS);
        }
    }

    vector<thread> bandthreads;
    vector<thread> fpgathreads;
    vector<thread> portthreads;

    printmutex.lock();
    cout << "Recording data from IP " << inipstr << endl;
    cout << "On ports: ";
    for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport)
        cout << *iport << ", ";
    cout << endl;
    printmutex.unlock();

    if (testtype == "f") {
        if (fpgavector.size() != portvector.size()) {
            cerr << "The number of FPGAs has to be the same as the number of ports!";
            exit(EXIT_FAILURE);
        }
        // NOTE: This test the percentage of data the specific FPGA receives
        for (int ifpga = 0; ifpga < fpgavector.size(); ifpga++) {
            fpgathreads.push_back(thread(TestFpga, portvector.at(ifpga), fpgavector.at(ifpga), inipstr, toread));
        }

        for(vector<thread>::iterator ithread = fpgathreads.begin(); ithread != fpgathreads.end(); ++ithread) {
            ithread -> join();
        }
    } else if (testtype == "p") {
        // NOTE: This tests the percentage of data the port receives
        for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport) {
            portthreads.push_back(thread(TestPort, *iport, inipstr, toread, recordstart));
        }

        for (vector<thread>::iterator ithread = portthreads.begin(); ithread != portthreads.end(); ++ithread) {
            ithread->join();
        }
    } else if (testtype == "w") {
        // NOTE: This tests all the 48 FPGAs at once
    } else {
        cerr << "Unrecognised option!";
        exit(EXIT_FAILURE);
    }
}
