#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <boost/array.hpp>
#include <boost/asio.hpp>
//#include <cufft.h>
#include <filterbank.hpp>
#include <vdif.hpp>

using std::cout;
using std::endl;

using boost::asio::ip::udp;

#define BUFLEN 8092
#define HEADER_SIZE 32

int main(int argc, char *argv[])
{
    try {

        int tmax = thread::hardware_concurrency();
        cout << "Can use a maximum of " << tmax << " threads.\n";

        unsigned char *inbuf = new unsigned char[BUFLEN];

        std::chrono::time_point<std::chrono::system_clock> send_begin, send_end;
        std::chrono::duration<double> send_elapsed;


        boost::asio::io_service io_service;
        udp::endpoint sender_endpoint;

        udp::socket socket(io_service, udp::v4());

        boost::asio::socket_base::reuse_address option(true);
        boost::asio::socket_base::receive_buffer_size option2(9000);
        socket.set_option(option);
        socket.set_option(option2);
        socket.bind(udp::endpoint(udp::v4(), atoi(argv[1])));

        cout << "Waiting to get something on port " << atoi(argv[1]) << " ..." << endl;

        size_t len;
        int n = 0;
        send_begin = std::chrono::system_clock::now();

        header_s heads;

        boost::array<unsigned char, BUFLEN> recv_buf;

        while(n < 1) {
            len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);
            cout << len << endl;
            get_header(recv_buf.data(), heads);
            n++;
        }

        std::ofstream dump("stream_dump.bin", std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        dump.write(reinterpret_cast<char*>(recv_buf.data() + HEADER_SIZE), BUFLEN - HEADER);
        dump.close();
        // fill it with some random data
        header_f headf;

        //save_filterbank(inbuf, headf);

        send_end = std::chrono::system_clock::now();
        send_elapsed = send_end - send_begin;

        cout << "Took " << send_elapsed.count() << " seconds to receive " << n << " buffers " << endl;

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
