#include <kernels.cuh>

__global__ void addtime(unsigned char *in, unsigned char *out, unsigned int jump, unsigned int factort)
{
    // index will tell which 1MHz channel we are taking care or
    // use 1 thread per 1MHz channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int ch = 0; ch < 27; ch++) {
        for (int t = 0; t < factort; t++) {
            out[0][idx * 27 + ch] += in[0][idx * 128 + ch + t * 32];
        }
    }

}

__global__ void addchannel(unsigned char *in, unsigned char *out, unsigned int factorc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int ch = 0; ch < factorc; ch++) {
        out[idx] += in[idx * factorc + ch];
    }

}

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump, unsigned int stream)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x + jump * 2 * stream;
	// offset introduced, jump to the B polarisation data - can cause some slowing down
	int idx2 = idx1 + jump;
    int saveidx = blockIdx.x * blockDim.x + threadsIdx.x + jump * stream;
    // these calculations assume polarisation is recorded in x,y base
    // i think the if statement is unnecessary as the number of threads for this
    // kernel 0s fftpoint * timeavg * nchans, which is exactly the size of the output array
    if (idx1 < jump) {      // half of the input data
        float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
        float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
        out[0][saveidx] = (power1 + power2); // I; what was this doing here? / 2.0;
        out[1][saveidx] = (power1 - power2); // Q
        out[2][saveidx] = 2 * in[idx1].x * in[idx2].x + 2 * in[idx1].y * in[idx2].y; // U
        out[3][saveidx] = 2 * in[idx1].x * in[idx2].y - 2 * in[idx1].y * in[idx2].x; // V
    }
}
