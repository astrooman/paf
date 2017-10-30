#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "config.hpp"
#include "gpu_pool.cuh"
#include "main_pool.cuh"

using std::move;
using std::thread;
using std::unique_ptr;
using std::vector;

MainPool::MainPool(InConfig config) : nogpus_(config.nogpus)
{
    for (int igpu = 0; igpu < nogpus_; igpu++) {
        gpuvector_.push_back(unique_ptr<GpuPool>(new GpuPool(igpu, config)));
    }

    for (int igpu = 0; igpu < nogpus_; igpu++) {
        threadvector_.push_back(thread(&GpuPool::Initialise, move(gpuvector_[igpu])));
    }
}

MainPool::~MainPool(void)
{
    for (int igpu = 0; igpu < nogpus_; igpu++) {
        threadvector_.at(igpu).join();
    }
}
