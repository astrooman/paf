#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
//#include <cufft.h>
#include "filterbank.hpp"
#include "pdif.hpp"

using std::cout;
using std::endl;

using boost::asio::ip::udp;

#define BUFLEN 8092
#define HEADER_SIZE 32
void GPUThread();
void ReceiveThread();

// add proper exception handling at some point
class Network
{
    public:
        Network(boost::asio::io_service &ios);
        ~Network(void){ netthread.join(); iothread.join();};
    protected:

    private:
        boost::array<unsigned char, BUFLEN> rec_buffer;
        void me_handler(const boost::system::error_code& error, std::size_t bytes_transferred, int gport);
        void receive(void);
        boost::asio::io_service ios_;
        udp::endpoint sender_endpoint;
        //udp::socket sock;
        //const int port;
        unsigned int ports_[6];
        //udp::socket *sockets_;
        std::vector<udp::socket> sockets;
        std::thread netthread;
        std::thread iothread;
        unsigned short count;
};

Network::Network(boost::asio::io_service &ios) : count(0)
{

    boost::asio::socket_base::reuse_address option(true);
    boost::asio::socket_base::receive_buffer_size option2(9000);        // creates false sense of receiving data in real time without this option

    //sockets_ = new udp::socket[6];

    for (int ii = 0; ii < 6; ii++) {

         sockets.push_back(udp::socket(ios, udp::endpoint(boost::asio::ip::address::from_string("10.17.0.2"), 17100 +ii)));
        //sockets.push_back(udp::socket(ios, udp::endpoint(udp::v4(), 17100 + ii)));
        sockets[ii].set_option(option);
        sockets[ii].set_option(option2);    

        cout << "Listening on port " << 17100 + ii << endl;

    }
    
    //sock.set_option(option);
    //sock.set_option(option2);

    cout << "Waiting to get something..." << endl;
    cout.flush();
    //receive();
    netthread = std::thread(&Network::receive, this);
    std::this_thread::sleep_for(std::chrono::seconds(1)); 
    iothread = std::thread([&ios]{ios.run();});
    //iothread.join();
}

void Network::receive(void)
{
    //std::this_thead::sleep_for (std::chrono::seconds(2));
    // fails badly if I try using std::bind
    // simply calling multiple sockets like that will block the application forever if nothing arrives at the associated port
    //sockets[0].async_receive_from(boost::asio::buffer(rec_buffer), sender_endpoint, boost::bind(&Network::me_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, 17100));
    sockets[3].async_receive_from(boost::asio::buffer(rec_buffer), sender_endpoint, boost::bind(&Network::me_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, 17102));
}

void Network::me_handler(const boost::system::error_code& error, std::size_t bytes_transferred, int gport)
{
    count++;
//    cout << "Grabbed packet " << count << " on port " << gport << endl;
//    cout << "In the handler" << endl;
//    cout << "Got " << bytes_transferred << "B" << endl;
    cout.flush();
    header_s heads;
    get_header(rec_buffer.data(), heads);

    size_t sum = 0;

    //for (int ii = 0; ii < 1000000000; ii++)
    //	sum++;


    if(count < 1024)
        receive();
}

int main(int argc, char *argv[])
{
    try {

        boost::asio::io_service ios;

        Network me_network(ios);
	std::thread thread1([&ios](){ios.run();});
        //thread1.join();
        //ios.run();
        cout << "Foo" << endl;
        cout.flush();
        cout << "Bar" << endl;
        cout.flush();
        thread1.join();
    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
