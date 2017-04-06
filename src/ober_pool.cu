#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "config.hpp"
#include "gpu_pool.cuh"
#include "ober_pool.cuh"

using std::move;
using std::thread;
using std::unique_ptr;
using std::vector;

/* ########################################################
TODO: Too many copies - could I use move in certain places?
#########################################################*/

Oberpool::Oberpool(InConfig config) : ngpus(config.ngpus)
{
    for (int ii = 0; ii < ngpus; ii++) {
        gpuvector.push_back(unique_ptr<GpuPool>(new GpuPool(ii, config)));
    }

    for (int ii = 0; ii < ngpus; ii++) {
        threadvector.push_back(thread(&GpuPool::execute, move(gpuvector[ii])));
    }
}

Oberpool::~Oberpool(void)
{
    for (int ii = 0; ii < ngpus; ii++) {
        threadvector[ii].join();
    }
}
