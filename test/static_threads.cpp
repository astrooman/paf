#include <iostream>
#include <thread>

struct my_struct
{
    int var1;
    int var2;
};

using std::cout;
using std::endl;
using std::thread;

void set_print(int ii)
{

    static my_struct tryme{5 * ii + 1,10 * ii + 2};
    cout << "I'm in thread " << ii << endl;
    cout << tryme.var1 << " " << tryme.var2 << endl;

    int stupid = ii * 25;

    cout << stupid << endl;

}
int main(int argc, char *argv[])
{

    thread* my_threads = new thread[5];

    for (int id = 0; id < 5; id++) {
        my_threads[id] = thread(set_print, id);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        my_threads[id].join();
    }

    return 0;
}
