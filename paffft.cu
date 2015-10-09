#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{



    const unsigned int arrsize = 32;
    const unsigned int fftsize = arrsize;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
    cufftComplex *h_inarray = new cufftComplex[arrsize];

    cout << "Pre-initialisation...\n";

    // this should make the first proper FFT execution faster
    cufftHanfle preinit;
    cufftPlan1d(&preinit, fftsize, CUFFT_C2C, 1);

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

    cufftHandle singleplan
    cufftPlan1d(&singleplan, fftsize, CUFFT_C2C, 1);
    cufftExecC2C(singleplan, d_inarray, d_inarray, CUFFT_FORWARD);


    return 0;

}
