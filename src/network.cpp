#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include "network.hpp"
#include "pdih.hpp"

using boost::asio::ip::udp;
using std::cout;
using std::endl;

#define BUFLEN 8092

Network::Network(boost::asio::io_servie &ios) : count(0)
{
    boost::asio::socket_base::reuse_address option1(true);
    boost::asio::socket_base::receive_buffer_size option2(8000);

    for (ii = 0; ii < 6; ii++) {
        sockets.push_back(udp::socket(ios, udp::endpoint(boost::asio::ip::address::from_string("xx.xx.xx.xx"), 17100 + ii)));
        sockets[ii].set_option(option1);
        sockets[ii].set_option(option2);
    }

    cout << "Waiting to receive something..." << endl;
    cout.flush();

    netthread = std::thread(&Network::receive, this);
    // sleeping makes sure that async_receive_from() was called before ios.run() call
    std::this_thread::sleep_for(std::chrono::seconds(1));   // need to sort this out properly later
    iothread = std::thread([&ios]{ios.run();});
}

void Network::receive(void)
{
    sockets[0].async_receive_from(boost::asio::buffer(rec_buffer), sender_endpoint, boost::bind(&Network::receive_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, 17100));
}

void Network::receive_handler(const boost::system::error_code &error, size_t bytes_transferred, int rport)
{
    count++;
    header_s heads;
    // possible race conditions here
    static obs_time start_time{head.epoch, head.ref_s};
    get_header(rec_buffer.data(), heads);

    if (count < 1024)
        receive();
}
