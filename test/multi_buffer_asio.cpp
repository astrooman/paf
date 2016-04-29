#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "filterbank.hpp"
#include "vdif.hpp"

using std::cout;
using std::endl;

using boost::asio::ip::udp;

#define BUFLEN 8092
#define HEADER_SIZE 32

class Network {

    public:
        Network(boost::asio::io_service &ios, int port) : ;
        ~Network() = default;

    protected:

    private:


}

int main(int argc, char *argv[])
{
    try {

        boost::asio::io_service ios;
        Network(ios, atoi(argv[1]));
        ios.run();

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }
    return 0;
}
