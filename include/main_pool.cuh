#ifndef _H_PAFRB_MAIN_POOL
#define _H_PAFRB_MAIN_POOL

/*! \file pool_multi.cuh
    \brief Defines classes that are responsible for all the work done

*/

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "config.hpp"
#include "gpu_pool.cuh"

class MainPool
{
    private:

        int nogpus_;

        std::vector<std::unique_ptr<GpuPool>> gpuvector_;
        std::vector<std::thread> threadvector_;
    protected:

    public:
        MainPool(void) = delete;
        MainPool(InConfig config);
        MainPool(const MainPool &inpool) = delete;
        MainPool& operator=(const MainPool &inpool) = delete;
        MainPool(MainPool &&inpool) = delete;
        MainPool& operator=(MainPool &&inpool) = delete;
        ~MainPool(void);
};

#endif
