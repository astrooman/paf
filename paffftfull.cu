/***********************************
VERSION OF THE FFT ALGORITHM FOR PAF
WITH FFT, POWER, AND AVERAGING ADDED

GENERAL TIMING RESULRS:
(E - EVENTS, P - NVPROF)


BOTH RESULTS QUOTED AS THERE ARE
SOME DISAGREEMENTS ON WHICH ONE IS
THE MOST RELIABLE ESTIMATE
***********************************/

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

    if (argc >= 2) {


    }


    const unsigned int arrsize = 32;
    const unsigned int fftsize = arrsize;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
    cufftComplex *h_inarray = new cufftComplex[arrsize];
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

    for (int ii = 0; ii < arrsize; ii++) {
            h_inarray[ii].x = arrdis(arreng);
            h_inarray[ii].y = arrdis(arreng);
    }

    cufftComplex *d_inarray;
    cudaMalloc((void**)&d_inarray, sizeof(cufftComplex) * arrsize);
    cudaMemcpy(d_inarray, h_inarray, sizeof(cufftComplex) * arrsize, cudaMemcpyHostToDevice);

    cout << "Performing single FFT...\n";

    cudaProfilerStart();

    PUSH_RANGE("Single FFT init", 1)
    cufftHandle singleplan;
    geterror(cufftPlan1d(&singleplan, fftsize, CUFFT_C2C, 1), "single FFT plan");
    POP_RANGE

    PUSH_RANGE("Single FFT exec", 2)
    geterror(cufftExecC2C(singleplan, d_inarray, d_inarray, CUFFT_FORWARD), "single FFT execution");
    POP_RANGE

    cufftDestroy(singleplan);
    cudaFree(d_inarray);
    delete [] h_inarray;

    cufftComplex *h_inarraym = new cufftComplex[arrsize * batchsize];

    for (int ii = 0; ii < arrsize * batchsize; ii++) {
        h_inarraym[ii].x = arrdis(arreng);
        h_inarraym[ii].y = arrdis(arreng);

    }

    cufftComplex *d_inarraym;
    cudaMalloc((void**)&d_inarraym, sizeof(cufftComplex) * arrsize * batchsize);
    cudaMemcpy(d_inarraym, h_inarraym, sizeof(cufftComplex) * arrsize * batchsize, cudaMemcpyHostToDevice);

    PUSH_RANGE("Multi FFT init", 3)
    cufftHandle multiplan;
    geterror(cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize), "multi FFT plan");
    POP_RANGE

    PUSH_RANGE("Multi FFT exec", 4)
    geterror(cufftExecC2C(multiplan, d_inarraym, d_inarraym, CUFFT_FORWARD), "multi FFT execution");
    POP_RANGE

    cufftDestroy(multiplan);
    cudaFree(d_inarraym);
    delete [] h_inarraym;

    unsigned int timesamp = 1;

    cufftComplex *h_inarraym2 = new cufftComplex[arrsize * batchsize * timesamp];

    for (int ii = 0; ii < arrsize * batchsize * timesamp; ii++) {
        h_inarraym2[ii].x = arrdis(arreng);
        h_inarraym2[ii].y = arrdis(arreng);

    }

    cufftComplex *d_inarraym2;
    cudaMalloc((void**)&d_inarraym2, sizeof(cufftComplex) * arrsize * batchsize * timesamp);
    cudaMemcpy(d_inarraym2, h_inarraym2, sizeof(cufftComplex) * arrsize * batchsize * timesamp, cudaMemcpyHostToDevice);

    PUSH_RANGE("Multi FFT 2 init", 5)
    cufftHandle multi2plan;
    geterror(cufftPlanMany(&multi2plan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "multi FFT 2 plan");
    POP_RANGE

    PUSH_RANGE("Multi FFT 2 exec", 6)
    geterror(cufftExecC2C(multi2plan, d_inarraym2, d_inarraym2, CUFFT_FORWARD), "multi FFT 2 execution");
    POP_RANGE

    cufftDestroy(multi2plan);
    cudaFree(d_inarraym2);
    delete [] h_inarraym2;

    cout << "Use different timing methods...\n";

    cufftComplex *h_inarraym3 = new cufftComplex[arrsize * batchsize * timesamp];

    for (int ii = 0; ii < arrsize * batchsize * timesamp; ii++) {
        h_inarraym3[ii].x = arrdis(arreng);
        h_inarraym3[ii].y = arrdis(arreng);

    }

    cufftComplex *d_inarraym3;
    cudaMalloc((void**)&d_inarraym3, sizeof(cufftComplex) * arrsize * batchsize * timesamp);
    cudaMemcpy(d_inarraym3, h_inarraym3, sizeof(cufftComplex) * arrsize * batchsize * timesamp, cudaMemcpyHostToDevice);

    cudaEvent_t init_start, init_end, exec_start, exec_end;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_end);
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_end);

    cudaEventRecord(init_start);
    cufftHandle multi3plan;
    geterror(cufftPlanMany(&multi3plan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "multi FFT 3 plan");
    cudaEventRecord(init_end);

    cudaEventRecord(exec_start);
    geterror(cufftExecC2C(multi3plan, d_inarraym3, d_inarraym3, CUFFT_FORWARD), "multi FFT 3 execution");
    cudaEventRecord(exec_end);

    float init_time, exec_time;

    cudaEventElapsedTime(&init_time, init_start, init_end);
    cudaEventElapsedTime(&exec_time, exec_start, exec_end);

    cout << "Init time: " << init_time << "ms\n";
    cout << "Exec time: " << exec_time << "ms\n";

    cufftDestroy(multi3plan);
    cudaFree(d_inarraym3);
    delete [] h_inarraym3;

    // try slightly different approach to memory to limit HtoD time
    cudaSetDeviceFlags(cudaDeviceMapHost);

    cufftComplex *h_inarraym4;
    cufftComplex *d_inarraym4;

    cudaHostAlloc((void**)&h_inarraym4, sizeof(cufftComplex) * arrsize * batchsize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&d_inarraym4, (void *)h_inarraym4, 0);

    for (int ii = 0; ii < arrsize * batchsize; ii++) {
        h_inarraym4[ii].x = arrdis(arreng);
        h_inarraym4[ii].y = arrdis(arreng);

    }

    cudaEventRecord(init_start);
    cufftHandle multi4plan;
    geterror(cufftPlanMany(&multi4plan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize), "multi FFT 4 plan");
    cudaEventRecord(init_end);

    cudaEventRecord(exec_start);
    geterror(cufftExecC2C(multi4plan, d_inarraym4, d_inarraym4, CUFFT_FORWARD), "multi FFT 4 execution");
    cudaEventRecord(exec_end);

    cudaEventElapsedTime(&init_time, init_start, init_end);
    cudaEventElapsedTime(&exec_time, exec_start, exec_end);

    cout << "Init time: " << init_time << "ms\n";
    cout << "Exec time: " << exec_time << "ms\n";

    cudaFreeHost(h_inarraym4);
    cufftDestroy(multi4plan);
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
