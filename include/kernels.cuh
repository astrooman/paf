#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

__global__ void addtime(unsigned char *in, unsigned char *out, unsigned int factort);

__global__ void addchannel(unsigned char *in, unsigned char *out, unsigned int factorc);

__global__ void powerscale(cufftComplex *in, unsigned char *out, unsigned int jump);

#endif
