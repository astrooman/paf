#include <chrono>
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
        std::chrono::time_point<std::chrono::system_clock> send_begin, send_end;
        std::chrono::duration<double> send_elapsed;

        boost::asio::socket_base::reuse_address option(true);
        boost::asio::io_service io_service;

        udp::endpoint sender_endpoint; // = *resolver.resolve(query);

        udp::socket socket(io_service, udp::v4());
        socket.set_option(option);
        socket.bind(udp::endpoint(udp::v4(), 45002));

        cout << "Waiting to get something..." << endl;

        boost::array<char, 8972> recv_buff;

        size_t len;
        int n = 0;
        send_begin = std::chrono::system_clock::now();

        while(n < 48 * 1000) {
            len = socket.receive_from(boost::asio::buffer(recv_buff), sender_endpoint);
            n++;
        }

        send_end = std::chrono::system_clock::now();
        send_elapsed = send_end - send_begin;

        cout << "Took " << send_elapsed.count() << " seconds to receive " << 1000 << " buffers " << endl;

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
