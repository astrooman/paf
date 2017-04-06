#ifndef _H_PAFRB_OBER_POOL
#define _H_PAFRB_OBER_POOL

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

class OberPool
{
    private:

        int ngpus;

        std::vector<std::unique_ptr<GpuPool>> gpuvector;
        std::vector<std::thread> threadvector;
    protected:

    public:
        OberPool(void) = delete;
        OberPool(InConfig config);
        OberPool(const OberPool &inpool) = delete;
        OberPool& operator=(const OberPool &inpool) = delete;
        OberPool(OberPool &&inpool) = delete;
        OberPool& operator=(OberPool &&inpool) = delete;
        ~OberPool(void);
        static void signal_handler(int signum);
};

#endif
