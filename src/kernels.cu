#include <kernels.cuh>

__global__ void poweradd(cufftComplex *in, unsigned char *out, unsigned int jump)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int idx2 = blockIdx.x * blockDim.x + threadIdx.x + jump;

    if (idx1 < jump) {      // half of the input data
        float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
        float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
        out[idx1] = (power1 + power2) / 2.0;
    }
}0
