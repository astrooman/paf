#include <stdio.h>

#include <kernels.cuh>

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

// __restrict__ tells the compiler there is no memory overlap

__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

__global__ void rearrange(cudaTextureObject_t texObj, cufftComplex * __restrict__ out)
{
    // this is currently the ugliest solution I can think of
    // xidx is the channel number
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int2 word;
    //if ((xidx == 0) && (yidx == 0)) printf("In the rearrange kernel\n");
    for (int sample = 0; sample < YSIZE; sample++) {
         word = tex2D<int2>(texObj, xidx, yidx + sample);
         printf("%i ", sample);
         out[xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
         out[xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
         out[336 * 128 + xidx * 128 + 7 * yidx + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
    }
}

__global__ void rearrange2(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc)
{

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int chanidx = threadIdx.x + blockIdx.y * 7;
    int skip;
    int2 word;

    for (int ac = 0; ac < acc; ac++) {
        skip = 336 * 128 * 2 * ac;
        for (int sample = 0; sample < YSIZE; sample++) {
            word = tex2D<int2>(texObj, xidx, yidx + ac * 48 * 128 + sample);
            out[skip + chanidx * YSIZE * 2 + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
        }
    }
}


__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort)
{

    // index will tell which 1MHz channel we are taking care or
    // use 1 thread per 1MHz channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx == 0) printf("In the time kernel\n");

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

/*__global__ void addtime(float* __restrict__ int, float* __restrict__ out, unsigned int jumpin, unsigned int jumpout, unsigned int factort)
{


} */

__global__ void addchannel(float* __restrict__ in, float* __restrict__ out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx == 0) printf("In the channel kernel\n");

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

__global__ void addchannel2(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize,  short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int extra = totsize - gulpno * gulp;
    // thats the starting save position for the chunk of length acc time samples
    int saveidx;

    int inskip;

    for (int ac = 0; ac < acc; ac++) {
        saveidx = (framet % (gulpno * gulp)) * nchans + idx;
        inskip = ac * 27 * 336;
        
        out[0][saveidx] = (float)0.0;
        out[1][saveidx] = (float)0.0;
        out[2][saveidx] = (float)0.0;
        out[3][saveidx] = (float)0.0;

        if ((framet % (gulpno * gulp)) >= extra) {
            for (int ch = 0; ch < factorc; ch++) {
                out[0][saveidx] += in[inskip + idx * factorc + ch];
                out[1][saveidx] += in[inskip + idx * factorc + ch + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ch + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ch + 3 * jumpin];
            }
        } else {
            for (int ch = 0; ch < factorc; ch++) {
                out[0][saveidx] += in[inskip + idx * factorc + ch];
                out[1][saveidx] += in[inskip + idx * factorc + ch + jumpin];
                out[2][saveidx] += in[inskip + idx * factorc + ch + 2 * jumpin];
                out[3][saveidx] += in[inskip + idx * factorc + ch + 3 * jumpin];
            }
            // save in two places -save in the extra bit 
            out[0][saveidx + (gulpno * gulp * nchans)] = out[0][saveidx];
            out[1][saveidx + (gulpno * gulp * nchans)] = out[1][saveidx];
            out[2][saveidx + (gulpno * gulp * nchans)] = out[2][saveidx];
            out[3][saveidx + (gulpno * gulp * nchans)] = out[3][saveidx];
            }
        framet++;
    }
    // not a problem - earch thread in a warp uses the same branch
/*    if ((framet % totsize) < gulpno * gulp) {
        for (int ac = 0; ac < acc; ac++) {
            inskip = ac * 27 * 336;
            outskip = ac * 27 * 336 / factorc;
            for (int ch = 0; ch < factorc; ch++) {
                out[0][outskip + saveidx] += in[inskip + idx * factorc + ch];
                out[1][outskip + saveidx] += in[inskip + idx * factorc + ch + jumpin];
                out[2][outskip + saveidx] += in[inskip + idx * factorc + ch + 2 * jumpin];
                out[3][outskip + saveidx] += in[inskip + idx * factorc + ch + 3 * jumpin];
            }
        }
    } else {
        for (int ac = 0; ac < acc; ac++) {
            for (int ch = 0; ch < factorc; ch++) {
                out[0][outskip + saveidx] += in[idx * factorc + ch];
                out[1][outskip + saveidx] += in[idx * factorc + ch + jumpin];
                out[2][outskip + saveidx] += in[idx * factorc + ch + 2 * jumpin];
                out[3][outskip + saveidx] += in[idx * factorc + ch + 3 * jumpin];
            }
            // save in two places - wrap wround to the start of the buffer
            out[0][outskip + saveidx - (gulpno * gulp * nchans)] = out[0][outskip + saveidx];
            out[1][outskip + saveidx - (gulpno * gulp * nchans)] = out[1][outskip + saveidx];
            out[2][outskip + saveidx - (gulpno * gulp * nchans)] = out[2][outskip + saveidx];
            out[3][outskop + saveidx - (gulpno * gulp * nchans)] = out[3][outskip + saveidx];
        }
    }
*/
}

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump)
{

    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx1 == 0) printf("In the power kernel\n");
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

__global__ void powertime(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort)
{
    // 1MHz channel ID
    int idx1 = blockIdx.x;
    // 'small' channel ID
    int idx2 = threadIdx.x;
    float power1;
    float power2;

    idx1 = idx1 * YSIZE * 2;
    int outidx = 27 * blockIdx.x + threadIdx.x;

    out[outidx] = (float)0.0;
    out[outidx + jump] = (float)0.0;
    out[outidx + 2 * jump] = (float)0.0;
    out[outidx + 3 * jump] = (float)0.0;

    for (int ii = 0; ii < factort; ii++) {
        idx2 = threadIdx.x + ii * 32;
	power1 = (in[idx1 + idx2].x * in[idx1 + idx2].x + in[idx1 + idx2].y * in[idx1 + idx2].y) * fftfactor;
        power2 = (in[idx1 + 128 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + 128 + idx2].y * in[idx1 + 128 + idx2].y) * fftfactor;
	out[outidx] += (power1 + power2);
        out[outidx + jump] += (power1 - power2);
        out[outidx + 2 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + idx2].y * in[idx1 + 128 + idx2].y));
        out[outidx + 3 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].y - in[idx1 + idx2].y * in[idx1 + 128 + idx2].x));

    }

   printf("%i, %i: %i\n", blockIdx.x, threadIdx.x, out[outidx]);
}

__global__ void powertime2(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc) {

    int idx1, idx2;
    int outidx;
    int skip1, skip2;
    float power1, power2;

    for (int ac = 0; ac < acc; ac++) {
        skip1 = ac * 336 * 128 * 2;
        skip2 = ac * 336 * 27;
        for (int ii = 0; ii < 7; ii++) {
            outidx = skip2 + 7 * 27 * blockIdx.x + ii * 27 + threadIdx.x;
            out[outidx] = (float)0.0;
            out[outidx + jump] = (float)0.0;
            out[outidx + 2 * jump] = (float)0.0;
            out[outidx + 3 * jump] = (float)0.0;

            idx1 = skip1 + 256 * (blockIdx.x * 7 + ii);

            for (int jj = 0; jj < factort; jj++) {
                idx2 = threadIdx.x + jj * 32;
                power1 = (in[idx1 + idx2].x * in[idx1 + idx2].x + in[idx1 + idx2].y * in[idx1 + idx2].y) * fftfactor;
                power2 = (in[idx1 + 128 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + 128 + idx2].y * in[idx1 + 128 + idx2].y) * fftfactor;
        	out[outidx] += (power1 + power2);
                out[outidx + jump] += (power1 - power2);
                out[outidx + 2 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + idx2].y * in[idx1 + 128 + idx2].y));
                out[outidx + 3 * jump] += (2 * fftfactor * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].y - in[idx1 + idx2].y * in[idx1 + 128 + idx2].x));
            }
        }
    }

//    printf("%i, %i: %i\n", blockIdx.x, threadIdx.x, out[outidx]);
}

__global__ void scale(float* in, float* out, unsigned int nchans, unsigned int time_samples)
{
    // call one block with 32 threads
    // be careful when processing total sizes that cannot be divided by 32
    // or make sure the total size can be divided by 32 when allocating
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float nrec = 1.0f / (float)nchans;
    float mean;
    float std;

    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int start = 0;
    float nrec1 = 1.0f / (float)(nchans -1.0f);
    for (int chunk = 0; chunk < (time_samples / threads); chunk++) {
        mean = 0.0f;
        std = 0.0f;
        start = chunk * threads * nchans;

        for (int ii = 0; ii < nchans; ii++) {
            mean += in[start + idx * nchans + ii] * nrec;
            //printf("%d\n", mean);
        }

        for (int jj = 0; jj < nchans; jj++) {
            std += (in[start + idx * nchans + jj] - mean) * (in[start + idx * nchans + jj] - mean);
        }
        std *= nrec1;

        //printf("%i: %i, %f, %f, %f\n", idx, nchans, nrec, mean, std);

        float stdrec = rsqrtf(std);

        for (int kk = 0; kk < nchans; kk++) {
            out[start + idx * nchans + kk] = ((in[start + idx * nchans + kk] - mean) * stdrec) * 32.0f + 64.0f;
            if (out[start + idx * nchans + kk] < 0.0f)
                out[start + idx * nchans + kk] = 0.0f;
        }
    }
}
