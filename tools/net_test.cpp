#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <tuple>
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

using std::cout;
using std::endl;
using std::mutex;
using std::string;
using std::thread;
using std::vector;

mutex printmutex;

#define SIZE 16384

void TestFpga(int iport, int nofpgas, std::string strip, int usebeam, int toread) {

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

    unsigned int framestamp{0};
    unsigned int timestamp{0};

    unsigned short beamno{0};
    unsigned short fpga{0};
    unsigned short refepoch{0};

    vector<std::tuple<unsigned short, unsigned short, unsigned short, unsigned int, unsigned int>> headvals;

    int ipack = 0;

    while (true) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        framestamp = (unsigned int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        if (framestamp == 0) {
            cout << "Reached the 27s boundary. Will start recording now...\n";
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
            headvals.push_back(std::make_tuple(fpga, beamno, refepoch, timestamp, framestamp));
            ipack++;
        }
    }



    printmutex.lock();
    cout << "Received on port " << iport << endl;
    for (int ifpga = 0; ifpga < 6; ifpga++) {
        cout << "FPGA " << iport * 6 + ifpga << ": " << fpgaid[ifpga] << endl;
    }
    cout << endl;
    printmutex.unlock();

    std::ofstream outhead("headers.codif", std::ios_base::out | std::ios_base::trunc);
    if (outhead) {
        for (auto ihead = headvals.begin(); ihead != headvals.end(); ++ihead) {
            outhead << std::get<0>(*ihead) << " " << std::get<1>(*ihead) << " "
                        << std::get<2>(*ihead) << " " << std::get<3>(*ihead) << " "
                        << std::get<4>(*ihead) << endl;
        }
    }
    outhead.close();

/*    std::ofstream outdata("data.codif", std::ios_base::out | std::ios_base::binary);
    if (outdata) {
        outdata.write(reinterpret_cast<char*>(inbuffer), (7168 + 64) * nofpgas * toread);
    }
    outdata.close();
*/
    //delete [] inbuffer;
    delete [] rec_buf;

}

void TestPort(int iport, std::string strip, unsigned int toread) {

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
    size_t frameno = 0 , highestframe = 0;
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    std::vector<size_t> framevals;

    while (true) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        frameno = (unsigned int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        if (frameno == 0) {
            cout << "Reached the 27s boundary. Will start recording now...\n";
            break;
        }
    }


    for (int ipack = 0; ipack < 8 * toread; ipack++) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;
        frameno = (size_t)(int)(rec_buf[7] | (rec_buf[6] << 8) | (rec_buf[5] << 16) | (rec_buf[4] << 24));
        framevals.push_back(frameno);	
        if (frameno > highestframe)
            highestframe = frameno;
    }

    printmutex.lock();
    cout << (double)(highestframe)/ (double)(toread - 1) * 100.0 << "% received on port " << iport << endl;
    cout << endl;
    printmutex.unlock();

    oss.str("");
    oss << "frames_port_" << iport;

    std::string portfile; 
    portfile = oss.str();
    std::ofstream outfile(portfile.c_str());

    for (auto iframe = framevals.begin(); iframe != framevals.end(); ++iframe)
        outfile << *iframe << std::endl;

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

    string fpgastr;
    string inipstr;
    string portstr;
    string testtype;

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
            nofpgas = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-s") {
            iarg++;
            toread = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-b") {
            iarg++;
            beam = atoi(argv[iarg]);
        } else if (string(argv[iarg]) == "-t") {
            iarg++;
            testtype = string(argv[iarg]);
        } else if (string(argv[iarg]) == "-h") {
            cout << "net_test [OPTIONS]\n"
                << "\t-i <ip>\n"
                << "\t-p <port1,port2,...,portn>\n"
                << "\t-s <packets to read>\n"
                << "\t-b <beam to use>\n"
                << "\t-t <p | b>\n\n";
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
/*
    if (testtype == "p") {
        for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport) {
            fpgathreads.push_back(thread(TestFpga, *iport, nofpgas, inipstr, beam, toread));
        }
        for (vector<thread>::iterator ithread = fpgathreads.begin(); ithread != fpgathreads.end(); ++ithread) {
            ithread->join();
        }
    } else if (testtype == "b") {
        // NOTE: Test the whole band
        unsigned int *fpgaframes = new unsigned int[portvector.size()];

        for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport) {
            bandthreads.push_back(thread(TestBand, *iport, inipstr, beam, toread, fpgaframes));
        }

        for (vector<thread>::iterator ithread = bandthreads.begin(); ithread != bandthreads.end(); ++ithread) {
            ithread->join();
        }

        for (int iport = 0; iport < portvector.size(); iport++) {
            cout << "Beam " << beam << " on port:\n"
                    << portvector[iport] << ": " << (float)fpgaframes[iport] / (float)toread * 100.0 << "%\n";
        }

        delete [] fpgaframes;
    }
*/
    // NOTE: this tests the percentage of data the port receives
    for (vector<int>::iterator iport = portvector.begin(); iport != portvector.end(); ++iport) {
        portthreads.push_back(thread(TestPort, *iport, inipstr, toread));
    }
    for (vector<thread>::iterator ithread = portthreads.begin(); ithread != portthreads.end(); ++ithread) {
        ithread->join();
    }

    return 0;
}
