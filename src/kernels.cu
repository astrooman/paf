#include <stdio.h>

#include "kernels.cuh"

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

#define INT_PER_LINE 2
#define NFPGAS 48
#define NCHAN_COARSE 336
#define NCHAN_FINE_IN 32
#define NCHAN_FINE_OUT 27
#define NACCUMULATE 128
#define NPOL 2
#define NSAMPS 4
#define NSAMPS_SUMMED 2
#define NCHAN_SUM 16
#define NSAMP_PER_PACKET 128
#define NCHAN_PER_PACKET 7

__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;


__global__ void UnpackKernel(int2 *__restrict__ in, cufftComplex *__restrict__ out) {

    int skip = 0;

    __shared__ int2 accblock[896];

    int chan = 0;
    int time = 0;
    int line = 0;

    cufftComplex cpol;
    int polint;

    int outskip = 0;

    for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
        // NOTE: This is skipping whole words as in will be cast to int2
        // skip = iacc * NCHAN_COARSE * NSAMP_PER_PACKET + blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        skip = blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        for (int ichunk = 0; ichunk < 7; ++ichunk) {
            line = ichunk * blockDim.x + threadIdx.x;
            chan = line % 7;
            time = line / 7;
            accblock[chan * NSAMP_PER_PACKET + time] = in[skip + line];
        }

        __syncthreads();

        skip = NCHAN_COARSE * NSAMP_PER_PACKET * NACCUMULATE;

        outskip = blockIdx.x * 7 * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET;

        for (chan = 0; chan < NCHAN_PER_PACKET; ++chan) {
            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].y;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));
            out[outskip + threadIdx.x] = cpol;

            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].x;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));

            out[skip + outskip + threadIdx.x] = cpol;

            outskip += NSAMP_PER_PACKET * NACCUMULATE;
        }
    }
}

__global__ void DetectScrunchKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans, short gulpno, size_t gulp, size_t extra, unsigned int framet)
{
  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes, but on the fine
   * channel read we perform an fft shift and exclude the band edges.
   */
  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;
  int pol_offset = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;
  int coarse_chan_offet = NACCUMULATE * NCHAN_FINE_IN * NSAMPS;
  int block_offset = NCHAN_FINE_IN * NSAMPS_SUMMED * blockIdx.x;
  int nwarps_per_block = blockDim.x/warpSize;


  //Here we calculate indexes for FFT shift.
  int offset_lane_idx = (lane_idx + 19)%32;

  //Here only first 27 lanes are active as we drop
  //5 channels due to the 32/27 oversampling ratio
  if (lane_idx < 27)
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)
      // This warp will loop over coarse channels in steps of NWARPS per block coarse_chan_idx (0,335)
      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int base_offset = coarse_chan_offet * coarse_chan_idx + block_offset + offset_lane_idx;

          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = base_offset + pol_offset * pol_idx;
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  // IDX = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE * pol_idx
                  // + NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx
                  // + blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED
                  // + NCHAN_FINE_IN * sample_idx
                  // + lane_idx;
                  cuComplex val = in[offset + (NCHAN_FINE_IN * sample_idx)]; // load frequencies in right order
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx] = real + imag;
            }
        }
    }

  __syncthreads();

  int saveoff = ((framet * 2 + blockIdx.x) % (gulpno * gulp)) * nchans;

  /**
   * Here each warp will reduce 32 channels into 2 channels
   * The last warp will have a problem that there will only be 16 values to process
   *
   */
  if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM))
    {
      float sum = 0.0;
      for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx)
        {
          sum += freq_sum_buffer[chan_idx];
        }
      out[saveoff + threadIdx.x] = sum;

      /**
       * Note [Ewan]: The code below is commented out as we turned off the
       * logic for handling the max_delay from the dedispersion. This can
       * and should be renabled if the max_delay logic is re-enabled
       */
      /*
      if (((framet * 2 + blockIdx.x) % (gulpno * gulp)) < extra) {
          out[saveoff + threadIdx.x + (gulpno * gulp) * nchans] = sum;
	  }*/
    }
  return;
}

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

__global__ void Transpose(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes) {

    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = in[idx + tsamp * nchans];
    }
}

__global__ void GetScaleFactors(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param) {
    // calculates mean and standard deviation in every channel
    // assumes the data has been transposed

    // for now have one thread per frequency channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mean;
    float variance;

    float ntrec = 1.0f / (float)ntimes;
    float ntrec1 = 1.0f / (float)(ntimes - 1.0f);

    unsigned int start = idx * ntimes;
    mean = 0.0f;
    variance = 0.0;
    // two-pass solution for now
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        mean += in[start + tsamp] * ntrec;
    }
    means[param][idx] = mean;

    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        variance += (in[start + tsamp] - mean) * (in[start + tsamp] - mean);
    }
    variance *= ntrec1;
    // reciprocal of standard deviation
    // multiplied by the desired standard deviation of the scaled data
    // reduces the number of operations that have to be done on the GPU
    rstdevs[param][idx] = rsqrtf(variance) * 32.0f;
    // to avoid inf when there is no data in the channel
    if (means[param][idx] == 0)
        rstdevs[param][idx] = 0;
}
