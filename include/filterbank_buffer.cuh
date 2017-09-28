#ifndef _H_PAFRB_FILTERBANK_BUFFER
#define _H_PAFRB_FILTERBANK_BUFFER

/*! \file buffer.cuh
    \brief Defines the main buffer class.

    This is the buffer that is used to aggregate the FFTed data before it is sent to the dedispersion.
    Uses a slightly convoluted version of a ring buffer (the same data chunk is occasionally saved into two places).
*/

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"

// NOTE: For the time being, only unsigned char scaled output will be supporded.
class FilterbankBuffer
{
    private:

        int accumulate_;
        int gpuid_;
        int nochans_;               // number of filterbank channels per time sample
        int nogulps_;               // number of gulps required in the buffer
        int nostokes_;              // number of Stokes parameters to keep in the buffer
        int typebytes_;

        std::mutex buffermutex_;
        std::mutex statemutex_;

        size_t end_;
        size_t extrasamples_;               // number of extra time samples required to process the full gulp
        size_t gulpsamples_;                // size of the single gulp
        size_t start_;
        size_t totalsamples_;               // total size of the data: nogulps_ * gulpsamples_ + extrasamples_


        ObsTime *gulptimes_;

        unsigned char **dfilterbank_;
        unsigned char **hdfilterbank_;
        unsigned char **rambuffer_;

        unsigned int *samplestate_;         // 0 for no data, 1 for data

        int filsaved_;
    protected:

    public:
        FilterbankBuffer(int gpuid);
        FilterbankBuffer(int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int gpuid);
        ~FilterbankBuffer(void);

        unsigned char **GetFilPointer(void) {return this -> hdfilterbank_;};

        int CheckIfReady(void);

        ObsTime GetTime(int index);

        void Allocate(int accumulate, int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int filchans, int stokesno, int filbits);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string outdir);
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void SendToRam(int idx, cudaStream_t &stream, int hostjump);
        void UpdateFilledTimes(ObsTime frame_time);
};
/*
// NOTE: Scaling will be removed in this form. Will be moved to the detection kernel
void FilterbankBuffer::GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs) {
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(float));
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        Transpose<<<1,nochans_,0,stream>>>(dfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_, d_transpose, nochans_, gulpsamples_ + extrasamples_);
        GetScaleFactors<<<1,nochans_,0,stream>>>(d_transpose, d_means, d_rstdevs, nochans_, gulpsamples_ + extrasamples_, istoke);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex_.lock();
    samplestate_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();

}
*/
#endif
