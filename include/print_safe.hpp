#ifndef _H_PAFRB_PRINT_SAFE
#define _H_PAFRB_PRINT_SAFE

#include <mutex>
#include <string>

std::recursive_mutex coutmutex;

template<class T> void PrintSafe(T lastin) {
    std::lock_guard<std::recursive_mutex> guard(coutmutex);
    std::cout << lastin << endl;
}

template<class T, class ... Types> void PrintSafe(T firstin, Types ... args) {
    std::lock_guard<std::recursive_mutex> guard(coutmutex);
    std::cout << firstin << " ";
    PrintSafe(args...);
}

#endif
