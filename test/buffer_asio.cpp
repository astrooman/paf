#include <iostream>

#include <boost/array.hpp>
#include <boost/asio.hpp>
//#include <cufft.h>
//#include <vdif.hpp>

using std::cout;
using std::endl;

using boost::asio::ip::udp;

int main(int argc, char *argv[])
{
    try {
        boost::asio::io_service io_service;

//        udp::resolver resolver(io_service);
//       udp::resolver::query query(udp::v4(), "192.168.2.100", "45002");

        udp::endpoint sender_endpoint; // = *resolver.resolve(query);

        udp::socket socket(io_service, udp::endpoint(udp::v4(), 45002));
//        socket.open(udp::v4());

        cout << "Waiting to get something..." << endl;

        boost::array<char, 16> recv_buff;
        size_t len = socket.receive_from(boost::asio::buffer(recv_buff), sender_endpoint);

        std::cout.write(recv_buff.data(), len);
    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
