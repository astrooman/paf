#include <cassert>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using std::condition_variable;
using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::thread;
using std::vector;

condition_variable condvar;
mutex condmut;

bool ready = false;
bool working = true;
queue<short*> myqueue;

void cond_waiting(int idx) {

    while (working) {
        std::unique_lock<mutex> condlock(condmut);
        condvar.wait(condlock, []{ return !myqueue.empty() || !working;});
        if (!myqueue.empty()) {
            short *localvar = myqueue.front();
            myqueue.pop();
            condlock.unlock();
            cout << "Thread " << idx << " can do work now" << endl;
            cout << "Thread " << idx << " received the following data: ";
            for (int idata = 0; idata < 4; idata++) {
                cout << *(localvar + idata) << " ";
            }
            cout << endl;
        }
    }
}

void prom_waiting(int idx) {

    while (working) {
        cout << "Thread " << idx << " can do work again" << endl;
    }
}

int main(int argc, char *argv[])
{

    vector<thread> mythreads;

    // 4 values per thread
    short *mydata = new short[4 * 8];
    cout << "This is what the full data looks like: ";
    for (int idata = 0; idata < 4 * 8; ++idata) {
        mydata[idata] = idata * 2 + 1;
        cout << mydata[idata] << " ";
    }

    cout << endl;

    for (int ithread = 0; ithread < 4; ithread++) {
        mythreads.push_back(thread(cond_waiting, ithread));
    }

    #pragma nounroll;
    for (int irep = 0; irep < 8; irep++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::lock_guard<mutex> condlock(condmut);
        cout << "The main thread is ready!" << endl;
        // NOTE: notify_one() will wakeup a random thread
        myqueue.push(mydata + irep * 4);
        condvar.notify_one();

    }

    working = false;
    condvar.notify_all();

    for (int ithread = 0; ithread < 4; ithread++) {
        mythreads.at(ithread).join();
    }

    mythreads.clear();
    assert(myqueue.empty());

    return 0;
}
