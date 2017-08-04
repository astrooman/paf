#include <chrono>
#include <string>
#include <vector>

class MyReceiver {
    public:

        MyReceiver(void) = delete;
        MyReceiver(std::string ipstr, std::vector<int> ports, std::chrono::system_clock::time_point start, int toread);
        ~MyReceiver(void);

        void DoWork(int idx);
        void GetData(int iport);
    private:

        int toread_;
        std::chrono::system_clock::time_point recordstart_;
        std::string ipstr_;
        std::vector<int> ports_;

    protected:
};
