#include <stdio.h>

#include <kernels.cuh>
__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

__global__ void rearrange(cudaTextureObject_t texObj, cufftComplex * __restrict__ out)
{
    // this is currently the ugliest solution I can think of
    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int2 word;

    for (int sample = 0; sample < YSIZE; sample++) {
         word = tex2D<int2>(texObj, xidx, yidx + sample);
         //printf("%i ", sample);
         out[xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
    }
}

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort)
{

    // index will tell which 1MHz channel we are taking care or
    // use 1 thread per 1MHz channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int ch = 0; ch < 27; ch++) {
	// have to restart to 0, otherwise will add to values from previous execution
        out[idx * 27 + ch] = (float)0.0;
        out[idx * 27 + ch + jumpout] = (float)0.0;
        out[idx * 27 + ch + 2 * jumpout] = (float)0.0;
        out[idx * 27 + ch + 3 * jumpout] = (float)0.0;

        for (int t = 0; t < factort; t++) {
            out[idx * 27 + ch] += in[idx * 128 + ch + t * 32];
            //printf("S1 time sum %f\n", out[idx * 27 + ch]);
            out[idx * 27 + ch + jumpout] += in[idx * 128 + ch + t * 32 + jumpin];
            out[idx * 27 + ch + 2 * jumpout] += in[idx * 128 + ch + t * 32 + 2 * jumpin];
            out[idx * 27 + ch + 3 * jumpout] += in[idx * 128 + ch + t * 32 + 3 * jumpin];
        }
    }
}

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    out[idx] = (float)0.0;
    out[idx + jumpout] = (float)0.0;
    out[idx + 2 * jumpout] = (float)0.0;
    out[idx + 3 * jumpout] = (float)0.0;

    for (int ch = 0; ch < factorc; ch++) {
        out[idx] += in[idx * factorc + ch];
        out[idx + jumpout] += in[idx * factorc + ch + jumpin];
        out[idx + 2 * jumpout] += in[idx * factorc + ch + 2 * jumpin];
        out[idx + 3 * jumpout] += in[idx * factorc + ch + 3 * jumpin];
    }

    //printf("S1 freq sum %f\n", out[idx]);
}

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump)
{

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced, jump to the B polarisation data - can cause some slowing down
	int idx2 = idx1 + jump;
    // these calculations assume polarisation is recorded in x,y base
    // i think the if statement is unnecessary as the number of threads for this
    // kernel 0s fftpoint * timeavg * nchans, which is exactly the size of the output array
    if (idx1 < jump) {      // half of the input data
        float power1 = (in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y) * fftfactor;
        float power2 = (in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y) * fftfactor;
        out[idx1] = (power1 + power2); // I; what was this doing here? / 2.0;
        //printf("Input numbers for %i and %i with jump %i: %f %f %f %f, with power %f\n", idx1, idx2, jump, in[idx1].x, in[idx1].y, in[idx2].x, in[idx2].y, out[idx1]);
        out[idx1 + jump] = (power1 - power2); // Q
        out[idx1 + 2 * jump] = 2 * fftfactor * (in[idx1].x * in[idx2].x + in[idx1].y * in[idx2].y); // U
        out[idx1 + 3 * jump] = 2 * fftfactor * (in[idx1].x * in[idx2].y - in[idx1].y * in[idx2].x); // V
    }
}
