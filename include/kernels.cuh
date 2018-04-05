#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void UnpackKernel(int2 *__restrict__ in, cufftComplex *__restrict__ out);

__global__ void DetectScrunchScaleKernel(cuComplex* __restrict__ in, float* __restrict__ out, float *means, float *scales, short nchans, short gulpno, size_t gulp, size_t extra, unsigned int framet);

__global__ void DetectScrunchKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans);

__global__ void GetScaleFactorsKernel(float *indata, float *base, float *stdev, float *factors, int nchans, int processed);

#endif
