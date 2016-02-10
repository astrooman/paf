#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

__global__ void poweradd(cufftComplex *in, unsigned char *out, unsigned int jump);

#endif
