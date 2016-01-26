#include <iostream>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <cufft.h>
#include <vdif.hpp>

using std::cout;
using std::endl;

using boost::asio::ip::udp;

int main(int argc, char *argv[])
{
    try {
        boost::asio::io_service io_service;

        udp::resolver resolver(io_service);
        udp::resolver::query query(udp::v4(), ,);

        udp::endpoint server_endpoint = *resolver.resolve(query);

        udp::socket socket(io_service);
        socket.open(udp::v4());

        boost::array<unsigned char, 16> recv_buff;
        size_t len = socket.receive_from(boost::asio::buffer(recv_buff), sender_endpoint);

        std::cout.write(recv_buff.data(), len);
    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
