#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using std::cout;
using std::endl;
using std::thread;
using std::vector;
using std::unique_ptr;

std::mutex cout_guard;

class GPUpool;

class Oberpool {

    public:
        Oberpool(void);
        Oberpool(const Oberpool &inpool) = delete;
        Oberpool& operator=(const Oberpool &inpool) = delete;
        Oberpool(Oberpool &&inpool) = delete;
        Oberpool& operator=(Oberpool &&inpool) = delete;
        ~Oberpool(void);
    private:
        //vector<GPUpool> gpuvector;
        //vector<GPUpool*> gpuvector;
        vector<unique_ptr<GPUpool>> gpuvector;
        vector<thread> threadvector;
};

class GPUpool {

    public:
        GPUpool(void);
        GPUpool(int ii);
        GPUpool(const GPUpool &inpool) = delete;
        GPUpool& operator=(const GPUpool &inpool) = delete;
        GPUpool(GPUpool &&inpool) { cout << "Move constructor " << endl; id = inpool.id; inpool.id = 0;};
        GPUpool& operator=(GPUpool &&inpool) = delete;
        ~GPUpool(void);
        void execute(void);
    private:
        double *buffer;
        int id;
};

Oberpool::Oberpool(void)
{
    cout << "Hello from Oberpool constructor!!" << endl;

    for (int ii = 0; ii < 1; ii++)
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii)));

    for (int ii = 0; ii < 1; ii++)
        threadvector.push_back(thread(&GPUpool::execute, std::move(gpuvector[ii])));

}

Oberpool::~Oberpool(void)
{
    cout_guard.lock();
    cout << "Bye from the Oberpool destructor!!" << endl;
    cout_guard.unlock();
    for (int ii = 0; ii < 1; ii++)
        threadvector[ii].join();

    cout << "Now properly bye from the Oberpool destructor!!" << endl;
}

GPUpool::GPUpool(void)
{
    cout << "Hello from the GPUpool constructor!!" << endl;
    cout.flush();
}

GPUpool::GPUpool(int ii)
{
    cout << "Hello from the GPUpool constructor with id " << ii << "!!" << endl;
    cout.flush();
    id = ii;
}

void GPUpool::execute(void)
{
    cout_guard.lock();
    cout << "Starting a GPU thread!!" << endl;
    cout_guard.unlock();
    buffer = new double[32];
    for (int ii = 0; ii < 32; ii++)
        buffer[ii] = id;
    std::this_thread::sleep_for(std::chrono::seconds(5));
}
GPUpool::~GPUpool(void)
{
    std::lock_guard<std::mutex> addguard(cout_guard);
    /*for (int ii = 0; ii < 32; ii++)
        cout << buffer[ii] << endl; */
    cout << "Bye fron the GPUpool destructor!!" << endl;
}

int main(int argc, char *argv[])
{
    Oberpool();
    return 0;
}
