#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

using std::cerr;
using std::cout;
using std::endl;


static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0x00fd482f };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();

void geterror(cufftResult res, std::string place);

int main(int argc, char* argv[])
{



    const unsigned int arrsize = 32;
    const unsigned int fftsize = arrsize;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
    int sizes[1] = {fftsize};


    cout << "Pre-initialisation...\n";

    PUSH_RANGE("FFT pre-init", 0)
    // this should make the first proper FFT execution faster
    cufftHandle preinit;
    cufftPlan1d(&preinit, fftsize, CUFFT_C2C, 1);
    POP_RANGE

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 arreng{seed};
    std::normal_distribution<float> arrdis(0.0, 1.0);

    cudaEvent_t init_start, init_end, exec_start, exec_end;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_end);
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_end);

    float init_time, exec_time;

    // try slightly different approach to memory to limit HtoD time
    cudaSetDeviceFlags(cudaDeviceMapHost);

    cufftComplex *h_inarraym;
    cufftComplex *d_inarraym;

    cudaHostAlloc((void**)&h_inarraym, sizeof(cufftComplex) * arrsize * batchsize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&d_inarraym, (void *)h_inarraym, 0);

    for (int ii = 0; ii < arrsize * batchsize; ii++) {
        h_inarraym[ii].x = arrdis(arreng);
        h_inarraym[ii].y = arrdis(arreng);

    }

    cudaEventRecord(init_start);
    cufftHandle multiplan;
    geterror(cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize), "multi FFT 4 plan");
    cudaEventRecord(init_end);

    cudaEventRecord(exec_start);
    geterror(cufftExecC2C(multiplan, d_inarraym, d_inarraym, CUFFT_FORWARD), "multi FFT 4 execution");
    cudaThreadSynchronize();
    cudaEventRecord(exec_end);

    cudaEventElapsedTime(&init_time, init_start, init_end);
    cudaEventElapsedTime(&exec_time, exec_start, exec_end);

    cout << "Init time: " << init_time << "ms\n";
    cout << "Exec time: " << exec_time << "ms\n";

    cudaFreeHost(h_inarraym);
    cufftDestroy(multiplan);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_end);
    cudaEventDestroy(exec_start);
    cudaEventDestroy(exec_end);

    cudaDeviceReset();

    return 0;

}

void geterror(cufftResult res, std::string place)
{
    if (res != CUFFT_SUCCESS)
        cout << "Error in " << place << "!! Error: " << res << endl;
}
