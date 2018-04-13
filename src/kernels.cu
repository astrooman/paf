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
#define NACCUMULATE 256
#define NPOL 2
#define NSAMPS 4
#define NSAMPS_SUMMED 2
#define NCHAN_SUM 16
#define NSAMP_PER_PACKET 128
#define NCHAN_PER_PACKET 7
#define INCHANS 567
#define OUTCHANS 512
#define SKIPCHAN 27

__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

__global__ void UnpackKernel(int2 *__restrict__ in, cufftComplex *__restrict__ out) {

    cufftComplex cpol;

    int chan = 0;
    int line = 0;
    int outskip = 0;
    int polint = 0;;
    int skip = 0;
    int time = 0;

    __shared__ int2 accblock[896];

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

__global__ void DetectScrunchScaleKernel(cuComplex* __restrict__ in, float* __restrict__ out, float *means, float *scales, short nchans, short gulpno, size_t gulp, size_t extra, unsigned int framet)
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

    int saveoff = ((framet * 2 + blockIdx.x) % (gulpno * gulp)) * OUTCHANS;

  /**
   * Here each warp will reduce 32 channels into 2 channels
   * The last warp will have a problem that there will only be 16 values to process
   *
   */
    // NOTE: Use 512 thread only, which will skip the first 28 averaged frequency channels
    // We are cutting out 55 channels - 28 from the bottom and 27 from the top of the band
    int skipbottom = 28 * NCHAN_SUM;
    if (threadIdx.x < OUTCHANS) {
    //if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM)) {
        float sum = 0.0;
        int scaled = 0;

        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[skipbottom + chan_idx];
        }
        // NOTE: Scaling factors are put in the 'right' filterbank order - with the highest frequency first
        scaled = __float2int_ru((sum - means[511 - threadIdx.x]) * scales[511 - threadIdx.x] + 64.5f);
        if (scaled > 255) {
            scaled = 255;
        } else if (scaled < 0) {
            scaled = 0;
        }
        //out[saveoff + threadIdx.x] = (unsigned char)scaled;
        // NOTE: That puts the highest frequency first (OUTCHANS - 1 - threadIdx.x)
        out[saveoff + 511 - threadIdx.x] = (unsigned char)scaled;

      /**
       * Note [Ewan]: The code below is commented out as we turned off the
       * logic for handling the max_delay from the dedispersion. This can
       * and should be renabled if the max_delay logic is re-enabled
       */
      /*
      if (((framet * 2 + blockIdx.x) % (gulpno * gulp)) < extra) {
          out[saveoff + threadIdx.x + (gulpno * gulp) * nchans] = scaled;
	  }*/
    }
    return;
}

__global__ void DetectScrunchKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans)
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

    int saveoff = blockIdx.x * OUTCHANS;
    int skipbottom = 28 * NCHAN_SUM;
    if (threadIdx.x < OUTCHANS) {
    //if ((threadIdx.x > SKIPCHAN) && (threadIdx.x < (INCHANS - SKIPCHAN))) {
    //if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM)) {
        float sum = 0.0;
        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[skipbottom + chan_idx];
        }
        //out[saveoff + threadIdx.x] = (unsigned char)scaled;
        // NOTE: That puts the highest frequency first (OUTCHANS - 1 - threadIdx.x)
        out[saveoff + 511 - threadIdx.x] = sum;
    }

    return;
}

__global__ void GetScaleFactorsKernel(float *indata, float *base, float *stdev, float *factors, int nchans, int processed) {
/*
    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float oldmean = base[threadIdx.x];

    //float estd = 0.0f;
    //float oldmean = 0.0;

    float val = 0.0f;
    float diff = 0.0;
    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    base[threadIdx.x] = mean;
    stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 2 * NACCUMULATE - 1.0f));
    // stdev[threadIdx.x] = estd;
*/
    float chmean = 0.0f;
    float chestd = 0.0f;
    float val = 0.0;
    float diff = 0.0;

    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - chmean;
        chmean += diff * factors[isamp + 1];
        chestd += diff * (val - chmean);
    }

    float oldmean = base[threadIdx.x];
    float oldestd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float newestd = 0.0f;

    diff = chmean - oldmean;
    base[threadIdx.x] = oldmean + diff * (float)(2.0f * NACCUMULATE) / (float)(processed + 2.0 * NACCUMULATE);
    newestd = oldestd + chestd + diff * diff * (float)(2.0f * NACCUMULATE) * (float)processed / (float)(processed + 2.0 * NACCUMULATE);
    stdev[threadIdx.x] = sqrt(newestd / (float)(processed + 2 * NACCUMULATE - 1.0f));

}
