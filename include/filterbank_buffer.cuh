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

template <class T>
class FilterbankBuffer
{
    private:
        vector<thrust::device_vector<float>> dfilterbank;              // stores different Stoke parameters
        vector<thrust::host_vector<float>> hfilterbank;                // stores Stokes parameters in the RAM buffer
        // TODO: Remove either phfilterbank or ph_fil
        float **pdfilterbank;                                          // array of raw pointers to Stoke parameters device vectors
        float **phfilterbank;                                          // same as above but for host vector
        float *ph_fil;
        size_t totsize;            // total size of the data: #gulps * gulp size + extra samples for dedispersion
        size_t gulp;            // size of the single gulp
        size_t extra;           // number of extra time samples required to process the full gulp
        int accumulate;
        int gpuid;
        int gulpno;             // number of gulps required in the buffer
        int nchans;             // number of filterbank channels per time sample
        int stokes;             // number of Stokes parameters to keep in the buffer
        int fil_saved;
        mutex buffermutex;
        mutex statemutex;
        size_t start;
        size_t end;
        ObsTime *gulptimes;
        T *d_buf;
        unsigned int *samplestate;     // 0 for no data, 1 for data
    protected:

    public:
        FilterbankBuffer(int id);
        FilterbankBuffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id);
        ~FilterbankBuffer(void);

        void Allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string outdir);
        float **GetFilPointer(void) {return this->pdfilterbank;};
        ObsTime GetTime(int index);
        int CheckIfReady();
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void SendToRam(unsigned char *out, int idx, cudaStream_t &stream, int host_jump);
        void UpdateFilledTimes(ObsTime frame_time);
        // add deleted copy, move, etc constructors
};

template<class T>
FilterbankBuffer<T>::FilterbankBuffer(int id) : gpuid(id) {
    cudaSetDevice(gpuid);
    start = 0;
    end = 0;
}

template<class T>
FilterbankBuffer<T>::FilterbankBuffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id) : extra(extra_u),
                                                                                gulp(gulp_u),
                                                                                gulpno(gulpno_u),
                                                                                totsize(size_u),
                                                                                gpuid(id) {
    start = 0;
    end = 0;
    //cudaMalloc((void**)&d_buf, totsize * sizeof(T));
    samplestate = new unsigned int[(int)totsize];
    std::fill(samplestate, samplestate + totsize, 0);
}

template<class T>
FilterbankBuffer<T>::~FilterbankBuffer() {
    end = 0;
}

template<class T>
void FilterbankBuffer<T>::Allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u) {
    fil_saved = 0;
    accumulate = acc_u;
    extra = extra_u;
    gulp = gulp_u;
    gulpno = gulpno_u;
    nchans = filchans;
    // size is the size of the buffer for the single Stokes parameter
    totsize = size_u;
    stokes = stokes_u;
    gulptimes = new ObsTime[gulpno];
    hfilterbank.resize(stokes);
    dfilterbank.resize(stokes);
    pdfilterbank = new float*[stokes];
    phfilterbank = new float*[stokes];
    for (int istoke = 0; istoke < stokes; istoke++) {
        // NOTE: Used to hold 2 full filterbank buffers
        hfilterbank[istoke].resize((gulp + extra) * 2 * nchans);
        phfilterbank[istoke] = thrust::raw_pointer_cast(hfilterbank[istoke].data());
        dfilterbank[istoke].resize(totsize * nchans);
        pdfilterbank[istoke] = thrust::raw_pointer_cast(dfilterbank[istoke].data());
    }
    cudaCheckError(cudaMalloc((void**)&d_buf, totsize * stokes * sizeof(T)));
    samplestate = new unsigned int[(int)totsize];
    cudaCheckError(cudaHostAlloc((void**)&ph_fil, (gulp + extra) * nchans * stokes * 2 * sizeof(float), cudaHostAllocDefault));
    std::fill(samplestate, samplestate + totsize, 0);
}

template<class T>
void FilterbankBuffer<T>::Deallocate(void) {
    cudaCheckError(cudaFreeHost(ph_fil));
    cudaCheckError(cudaFree(d_buf));
    delete [] samplestate;
    delete [] gulptimes;
    delete [] pdfilterbank;
    delete [] phfilterbank;
}

template<class T>
void FilterbankBuffer<T>::SendToDisk(int idx, header_f header, std::string outdir) {
        SaveFilterbank(ph_fil, gulp + extra, (gulp + extra) * nchans * stokes * idx, header, stokes, fil_saved, outdir);
        fil_saved++;
}

template<class T>
int FilterbankBuffer<T>::CheckIfReady() {
    std::lock_guard<mutex> addguard(statemutex);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < gulpno; igulp++) {
        if (samplestate[(igulp + 1) * gulp + extra - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

template<class T>
void FilterbankBuffer<T>::GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs) {
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulp + extra) * nchans * sizeof(float));
    for (int ii = 0; ii < stokes; ii++) {
        Transpose<<<1,nchans,0,stream>>>(pdfilterbank[ii] + (idx - 1) * gulp * nchans, d_transpose, nchans, gulp + extra);
        GetScaleFactors<<<1,nchans,0,stream>>>(d_transpose, d_means, d_rstdevs, nchans, gulp + extra, ii);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex.lock();
    samplestate[idx * gulp + extra - 1] = 0;
    statemutex.unlock();
}


template<class T>
void FilterbankBuffer<T>::SendToRam(unsigned char *out, int idx, cudaStream_t &stream, int host_jump) {
    // which half of the RAM buffer we are saving into
    host_jump *= (gulp + extra) * nchans * stokes;
    // dump to the host memory only - not interested in the dedisperion in the dump mode
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump, pdfilterbank[0] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 1 * (gulp + extra) * nchans, pdfilterbank[1] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 2 * (gulp + extra) * nchans, pdfilterbank[2] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 3 * (gulp + extra) * nchans, pdfilterbank[3] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    statemutex.lock();

    samplestate[idx * gulp + extra - 1] = 0;
    statemutex.unlock();
}

template<class T>
void FilterbankBuffer<T>::UpdateFilledTimes(ObsTime frame_time) {
    std::lock_guard<mutex> addguard(statemutex);
    int framet = frame_time.framefromstart;
    int index = frame_time.framefromstart % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();

    for (int ii = 0; ii < accumulate; ii++) {
        index = framet % (gulpno * gulp);
        if((index % gulp) == 0)
            gulptimes[index / gulp] = frame_time;
        samplestate[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        if ((index < extra) && (framet > extra)) {
            samplestate[index + gulpno * gulp] = 1;
        }
        framet++;
    }
}

template<class T>
ObsTime FilterbankBuffer<T>::GetTime(int index) {
    return gulptimes[index];
}

#endif
