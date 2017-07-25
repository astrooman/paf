#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void RearrangeKernel(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc);

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort);

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc);

__global__ void addchannel2(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc);

__global__ void AddChannelsScaleKernel(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc, float **means, float **rstdevs);

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump);

__global__ void powertime(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort);

__global__ void GetPowerAddTimeKernel(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc);

__global__ void InitScaleFactors(float **means, float **rstdevs, int stokes);

__global__ void Transpose(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes);

__global__ void GetScaleFactors(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

// NOTE: New kernels
__global__ void UnpackKernel(int2 *__restrict__ in, cufftComplex *__restrict__ out);

__global__ void DetectScrunchKernel(cuComplex* __restrict__ in, float** __restrict__ out, short nchans, short gulpno, size_t gulp, size_t extra, unsigned int framet);

// TODO: This will become the kernel to calculate scaling factors
__global__ void bandpass();
#endif
