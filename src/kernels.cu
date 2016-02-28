#include <kernels.cuh>

__global__ void addtime(unsigned char *in, unsigned char *out, unsigned int factort)
{
    // index will tell which 1MHz channel we are taking care or
    // use 1 thread per 1MHz channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int ch = 0; ch < 27; ch++) {
        for (int t = 0; t < factort; t++) {
            out[idx * 27 + ch] += in[idx * 128 + ch + t * 32];
        }
    }

}

__global__ void addchannel(unsigned char *in, unsigned char *out, unsigned int factorc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int ch = 0; ch < factorc; ch++) {
        out[idx] += in[idx * factorc + ch];
    }

}

__global__ void powerscale(cufftComplex *in, unsigned char *out, unsigned int jump)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int idx2 = idx1 + jump;

    if (idx1 < jump) {      // half of the input data
        float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
        float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
        out[idx1] = (power1 + power2) / 2.0;
    }
}0
