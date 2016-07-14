#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void rearrange(cudaTextureObject_t texObj, cufftComplex * __restrict__ out);
__global__ void rearrange2(cudaTextureObject_t texObj, cufftComplex * __restrict__ out);

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort);

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc);

__global__ void addchannel2(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet);

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump);

__global__ void powertime(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort);

__global__ void powertime2(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort);
#endif
