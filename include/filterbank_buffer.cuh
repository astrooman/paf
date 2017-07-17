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
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"

using std::mutex;
using std::vector;

template <class BufferType>
class FilterbankBuffer
{
    private:

        int accumulate_;
        int gpuid_;
        int nochans_;               // number of filterbank channels per time sample
        int nogulps_;               // number of gulps required in the buffer
        int nostokes_;              // number of Stokes parameters to keep in the buffer

        mutex buffermutex_;
        mutex statemutex_;

        size_t end_;
        size_t extrasamples_;               // number of extra time samples required to process the full gulp
        size_t gulpsamples_;                // size of the single gulp
        size_t start_;
        size_t totalsamples_;               // total size of the data: nogulps_ * gulpsamples_ + extrasamples_

        unsigned int *samplestate_;         // 0 for no data, 1 for data

        ObsTime *gulptimes_;

        BufferType *dfilterbank_
        BufferType **hdfilterbank_;                                          // array of raw pointers to Stoke parameters device vectors
        BufferType **rambuffer_;
        
        int fil_saved;
    protected:

    public:
        FilterbankBuffer(int gpuid);
        FilterbankBuffer(int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int gpuid);
        ~FilterbankBuffer(void);

        BufferType **GetFilPointer(void) {return this->dfilterbank_;};

        int CheckIfReady(void);

        ObsTime GetTime(int index);

        void Allocate(int accumulate, int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int filchans, int stokesno);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string outdir);
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void SendToRam(int idx, cudaStream_t &stream, int hostjump);
        void UpdateFilledTimes(ObsTime frame_time);
};

template<class BufferType>
FilterbankBuffer<BufferType>::FilterbankBuffer(int gpuid) : gpuid_(gpuid) {
    cudaSetDevice(gpuid_);
    start_ = 0;
    end_ = 0;
}

template<class BufferType>
FilterbankBuffer<BufferType>::FilterbankBuffer(int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int gpuid) : extrasamples_(extrasize),
                                                                                gulpsamples_(gulpsize),
                                                                                nogulps_(gulpno),
                                                                                totalsamples_(totalsize),
                                                                                gpuid_(gpuid) {
    start_ = 0;
    end_ = 0;
    samplestate_ = new unsigned int[(int)totalsamples_];
    std::fill(samplestate_, samplestate_ + totalsamples_, 0);
}

template<class BufferType>
FilterbankBuffer<BufferType>::~FilterbankBuffer() {
    end_ = 0;
}

template<class BufferType>
void FilterbankBuffer<BufferType>::Allocate(int accumulate, int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int filchans, int stokesno) {
    fil_saved = 0;
    accumulate_ = accumulate;
    extrasamples_ = extrasize;
    gulpsamples_ = gulpsize;
    nogulps_ = gulpno;
    nochans_ = filchans;
    totalsamples_ = totalsize;
    nostokes_ = stokesno;

    gulptimes_ = new ObsTime[nogulps_];
    hdfilterbank = new float*[nostokes_];
    samplestate_ = new unsigned int[(int)totalsamples_];
    std::fill(samplestate_, samplestate_ + totalsamples_, 0);
    cudaCheckError(cudaHostAlloc((void**)&rambuffer_, nostokes_ * sizeof(BufferType*), cudaHostAllocDefault));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&(hdfilterbank_[istoke]), totalsamples_ * nochans_ * sizeof(BufferType)));
        cudaCheckError(cudaHostAlloc((void**)&(rambuffer_[istoke]), (gulpsamples_ + extrasamples_) * nochans_ * 2 * sizeof(BufferType), cudaHostAllocDefault));
    }

    cudaCheckError(cudaMalloc((void**)&dfilterbank_, nostokes_ * sizeof(BufferType*)));
    cudaCheckError(cudaMemcpy(dfilterbank_, hdfilterbank_, nostokes_ * sizeof(BufferType*), cudaMemcpyHostToDevice));
}

template<class BufferType>
void FilterbankBuffer<BufferType>::Deallocate(void) {

    cudaCheckError(cudaFree(dfilterbank_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFreeHost(rambuffer_[istoke]));
        cudaCheckError(cudaFree(hdfilterbank_[istoke]));
    }

    cudaCheckError(cudaFreeHost(rambuffer_));

    delete [] gulptimes_;
    delete [] hdfilterbank_;
    delete [] samplestate_;
}

template<class BufferType>
void FilterbankBuffer<BufferType>::SendToDisk(int idx, header_f header, std::string outdir) {
        SaveFilterbank(rambuffer_, gulpsamples_ + extrasamples_, (gulpsamples_ + extrasamples_) * nochans_ * nostokes_ * idx, header, nostokes_, fil_saved, outdir);
        fil_saved++;
}

template<class BufferType>
int FilterbankBuffer<BufferType>::CheckIfReady() {
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < nogulps_; igulp++) {
        if (samplestate_[(igulp + 1) * gulpsamples_ + extrasamples_ - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

template<class BufferType>
void FilterbankBuffer<BufferType>::GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs) {
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(float));
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        Transpose<<<1,nochans_,0,stream>>>(pdfilterbank[istoke] + (idx - 1) * gulpsamples_ * nochans_, d_transpose, nochans_, gulpsamples_ + extrasamples_);
        GetScaleFactors<<<1,nochans_,0,stream>>>(d_transpose, d_means, d_rstdevs, nochans_, gulpsamples_ + extrasamples_, istoke);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex_.lock();
    samplestate_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}


template<class BufferType>
void FilterbankBuffer<BufferType>::SendToRam(int idx, cudaStream_t &stream, int hostjump) {
    // NOTE: Which half of the RAM buffer we are saving into
    hostjump *= (gulpsamples_ + extrasamples_) * nochans_ * nostokes_;

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMemcpyAsyc(rambuffer_[istoke] + hostjump, hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType), cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);
    statemutex_.lock();
    samplestate_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}

template<class BufferType>
void FilterbankBuffer<BufferType>::UpdateFilledTimes(ObsTime frame_time) {
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frame_time.framefromstart;
    int index = frame_time.framefromstart % (nogulps_ * gulpsamples_);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();

    for (int iacc = 0; iacc < accumulate_; iacc++) {
        index = framet % (nogulps_ * gulpsamples_);
        if((index % gulpsamples_) == 0)
            gulptimes_[index / gulpsamples_] = frame_time;
        samplestate_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totalsamples_ << std::endl;
        //std::cout.flush();
        if ((index < extrasamples_) && (framet > extrasamples_)) {
            samplestate_[index + nogulps_ * gulpsamples_] = 1;
        }
        framet++;
    }
}

template<class BufferType>
ObsTime FilterbankBuffer<BufferType>::GetTime(int index) {
    return gulptimes_[index];
}

#endif
