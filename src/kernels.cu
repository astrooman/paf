#include <stdio.h>

#include "kernels.cuh"

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

// NOTE: Initialise the scaling factors
// Use custom kernel as CUDA memset is slower and not safe for anything else than int
__global__ void InitScaleFactors(float **means, float **rstdevs, int stokes) {
    // NOTE: The scaling is (in - mean) * rstdev + 64.0f
    // I want to get the original in back in the first running
    // Will therefore set the mean to 64.0f and rstdev to 1.0f

    // NOTE: Each thread responsible for one channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int istoke = 0; istoke < stokes; istoke++) {
        means[istoke][idx] = 64.0f;
        rstdevs[istoke][idx] = 1.0f;
    }
}

__global__ void RearrangeKernel(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc)
{

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int chanidx = threadIdx.x + blockIdx.y * 7;
    int skip;
    int2 word;

    for (int iac = 0; iac < acc; iac++) {
        skip = 336 * 128 * 2 * iac;
        for (int sample = 0; sample < YSIZE; sample++) {
            word = tex2D<int2>(texObj, xidx, yidx + iac * 48 * 128 + sample);
            out[skip + chanidx * YSIZE * 2 + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
        }
    }
}

__global__ void GetPowerAddTimeKernel(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc) {

    int idx1, idx2;
    int outidx;
    int skip1, skip2;
    float power1, power2;
    float avgfactor= 1.0f / factort;

    for (int iac = 0; iac < acc; iac++) {
        skip1 = iac * 336 * 128 * 2;
        skip2 = iac * 336 * 27;
            for (int ichan = 0; ichan < 7; ichan++) {
            outidx = skip2 + 7 * 27 * blockIdx.x + ichan * 27 + threadIdx.x;
            out[outidx] = (float)0.0;
            out[outidx + jump] = (float)0.0;
            out[outidx + 2 * jump] = (float)0.0;
            out[outidx + 3 * jump] = (float)0.0;

            idx1 = skip1 + 256 * (blockIdx.x * 7 + ichan);

            for (int itime = 0; itime < factort; itime++) {
                idx2 = threadIdx.x + itime * 32;
                power1 = (in[idx1 + idx2].x * in[idx1 + idx2].x + in[idx1 + idx2].y * in[idx1 + idx2].y) * fftfactor;
                power2 = (in[idx1 + 128 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + 128 + idx2].y * in[idx1 + 128 + idx2].y) * fftfactor;
                out[outidx] += (power1 + power2) * avgfactor;
                out[outidx + jump] += (power1 - power2) * avgfactor;
                out[outidx + 2 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + idx2].y * in[idx1 + 128 + idx2].y)) * avgfactor;
                out[outidx + 3 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].y - in[idx1 + idx2].y * in[idx1 + 128 + idx2].x)) * avgfactor;
            }
        }
    }
}

__global__ void AddChannelsKernel(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize,  short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int extra = totsize - gulpno * gulp;
    // thats the starting save position for the chunk of length acc time samples
    int saveidx;

    int inskip;

    for (int iac = 0; iac < acc; iac++) {
        saveidx = (framet % (gulpno * gulp)) * nchans + idx;
        inskip = iac * 27 * 336;

        out[0][saveidx] = (float)0.0;
        out[1][saveidx] = (float)0.0;
        out[2][saveidx] = (float)0.0;
        out[3][saveidx] = (float)0.0;

        if ((framet % (gulpno * gulp)) >= extra) {
            for (int ichan = 0; ichan < factorc; ichan++) {
                out[0][saveidx] += in[inskip + idx * factorc + ichan];
                out[1][saveidx] += in[inskip + idx * factorc + ichan + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ichan + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ichan + 3 * jumpin];
            }
        } else {
            for (int ichan = 0; ichan < factorc; ichan++) {
                out[0][saveidx] += in[inskip + idx * factorc + ichan];
                out[1][saveidx] += in[inskip + idx * factorc + ichan + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ichan + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ichan + 3 * jumpin];
            }
            // save in two places -save in the extra bit
            out[0][saveidx + (gulpno * gulp * nchans)] = out[0][saveidx];
            out[1][saveidx + (gulpno * gulp * nchans)] = out[1][saveidx];
            out[2][saveidx + (gulpno * gulp * nchans)] = out[2][saveidx];
            out[3][saveidx + (gulpno * gulp * nchans)] = out[3][saveidx];
            }
        framet++;
    }
}

__global__ void AddChannelsScaleKernel(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize,  short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc, float **means, float **rstdevs) {

    // NOTE: The number of threads is equal to the number of output channels
    // Each 'idx' is responsible for one output frequency channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int extra = totsize - gulpno * gulp;
    float avgfactor = 1.0f / factorc;
    // thats the starting save position for the chunk of length acc time samples
    int saveidx;

    float tmp0, tmp1, tmp2, tmp3;

    int inskip;

    for (int iac = 0; iac < acc; iac++) {
        // NOTE: Channels in increasing order
        // saveidx = (framet % (gulpno * gulp)) * nchans + idx;
        // channels in decreasing order
        saveidx = (framet % (gulpno * gulp)) * nchans + nchans - (idx + 1);
        inskip = iac * 27 * 336;

        out[0][saveidx] = (float)0.0;
        out[1][saveidx] = (float)0.0;
        out[2][saveidx] = (float)0.0;
        out[3][saveidx] = (float)0.0;

        // NOTE: Use scaling of the form
        // out = (in - mean) / stdev * 32 + 64;
        // rstdev = (1 / stdev) * 32 to reduce the number of operations
        if ((framet % (gulpno * gulp)) >= extra) {
            // NOTE: Save in one place in the filterbank buffer
            for (int ichan = 0; ichan < factorc; ichan++) {
                out[0][saveidx] += in[inskip + idx * factorc + ichan];
                out[1][saveidx] += in[inskip + idx * factorc + ichan + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ichan + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ichan + 3 * jumpin];
            }
            // scaling
            out[0][saveidx] = (out[0][saveidx] * avgfactor - means[0][idx]) * rstdevs[0][idx] + 64.0f;
            out[1][saveidx] = (out[1][saveidx] * avgfactor - means[1][idx]) * rstdevs[1][idx] + 64.0f;
            out[2][saveidx] = (out[2][saveidx] * avgfactor - means[2][idx]) * rstdevs[2][idx] + 64.0f;
            out[3][saveidx] = (out[3][saveidx] * avgfactor - means[3][idx]) * rstdevs[3][idx] + 64.0f;
        } else {
            // NOTE: Save in two places in the filterbank buffer
            for (int ichan = 0; ichan < factorc; ichan++) {
                out[0][saveidx] += in[inskip + idx * factorc + ichan];
                out[1][saveidx] += in[inskip + idx * factorc + ichan + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ichan + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ichan + 3 * jumpin];
            }

            out[0][saveidx] = (out[0][saveidx] * avgfactor - means[0][idx]) * rstdevs[0][idx] + 64.0f;
            out[1][saveidx] = (out[1][saveidx] * avgfactor - means[1][idx]) * rstdevs[1][idx] + 64.0f;
            out[2][saveidx] = (out[2][saveidx] * avgfactor - means[2][idx]) * rstdevs[2][idx] + 64.0f;
            out[3][saveidx] = (out[3][saveidx] * avgfactor - means[3][idx]) * rstdevs[3][idx] + 64.0f;
            tmp0 = rintf(fminf(fmaxf(0.0, out[0][saveidx]), 255.0));
            out[0][saveidx] = tmp0;
            //out[0][saveidx] = fminf(255, out[0][saveidx]);
            out[1][saveidx] = fmaxf(0.0, out[0][saveidx]);
            out[1][saveidx] = fminf(255, out[0][saveidx]);
            out[2][saveidx] = fmaxf(0.0, out[0][saveidx]);
            out[2][saveidx] = fminf(255, out[0][saveidx]);
            out[3][saveidx] = fmaxf(0.0, out[0][saveidx]);
            out[3][saveidx] = fminf(255, out[0][saveidx]);

            out[0][saveidx + (gulpno * gulp * nchans)] = out[0][saveidx];
            out[1][saveidx + (gulpno * gulp * nchans)] = out[1][saveidx];
            out[2][saveidx + (gulpno * gulp * nchans)] = out[2][saveidx];
            out[3][saveidx + (gulpno * gulp * nchans)] = out[3][saveidx];
        }
        framet++;
    }

}
