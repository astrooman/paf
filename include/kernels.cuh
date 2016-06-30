#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void rearrange(cudaTextureObject_t texObj, cufftComplex * __restrict__ out);

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort);

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc);

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump);

#endif
