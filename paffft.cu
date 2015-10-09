#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

using std::cerr;
using std::cout;
using std::endl;


static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
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

int main(int argc, char* argv[])
{



    const unsigned int arrsize = 32;
    const unsigned int fftsize = arrsize;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
    cufftComplex *h_inarray = new cufftComplex[arrsize];


    cout << "Pre-initialisation...\n";

    PUSH_RANGE("FFT pre-init", 0)
    // this should make the first proper FFT execution faster
    cufftHandle preinit;
    cufftPlan1d(&preinit, fftsize, CUFFT_C2C, 1);
    POP_RANGE

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 arreng{seed};
    std::normal_distribution<float> arrdis(0.0, 1.0);

    for (int ii = 0; ii < arrsize; ii++) {
            h_inarray[ii].x = arrdis(arreng);
            h_inarray[ii].y = arrdis(arreng);
    }

    cufftComplex *d_inarray = new cufftComplex[arrsize];
    cudaMalloc((void**)&d_inarray, sizeof(cufftComplex) * arrsize);
    cudaMemcpy(d_inarray, h_inarray, sizeof(cufftComplex) * arrsize, cudaMemcpyHostToDevice);

    cout << "Performing single FFT...\n";

    PUSH_RANGE("Single FFT init", 1)
    cufftHandle singleplan;
    cufftPlan1d(&singleplan, fftsize, CUFFT_C2C, 1);
    POP_RANGE

    PUSH_RANGE("Single FFT exec", 2)
    cufftExecC2C(singleplan, d_inarray, d_inarray, CUFFT_FORWARD);
    POP_RANGE

    cufftDestroy(singleplan);
    cudaFree(d_inarray);
    delete [] h_inarray;

    cufftComplex *h_inarraym = new cufftComplex[arrsize * batchsize];

    for (int ii = 0; ii < arrsize * batchsize; ii++) {
        h_inarraym[ii].x = arrdis(arreng);
        h_inarraym[ii].y = arrdis(arreng);

    }

    cufftComplex *d_inarraym = new cufftComplex[arrsize * batchsize];
    cudaMalloc((void**)&d_inarraym, sizeof(cufftComplex) * arrsize * batchsize);
    cudaMemcpy(d_inarraym, h_inarraym, sizeof(cufftComplex) * arrsize * batchsize, cudaMemcpyHostToDevice);

    cout << "Performing multi FFT...\n";

    int sizes[1] = {fftsize};

    PUSH_RANGE("Multi FFT init", 3)
    cufftHandle multiplan;
    cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize);
    POP_RANGE

    PUSH_RANGE("Multi FFT exec", 4)
    cufftExecC2C(multiplan, d_inarraym, d_inarraym, CUFFT_FORWARD);
    POP_RANGE

    cufftDestroy(multiplan);
    cudaFree(d_inarraym);
    delete [] h_inarraym;

    return 0;

}
