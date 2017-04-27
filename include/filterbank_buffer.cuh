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
        vector<thrust::device_vector<float>> d_filterbank;              // stores different Stoke parameters
        vector<thrust::host_vector<float>> h_filterbank;                // stores Stokes parameters in the RAM buffer
        float **pd_filterbank;                                          // array of raw pointers to Stoke parameters device vectors
        float **ph_filterbank;                                          // same as above but for host vector
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
        ObsTime *gulp_times;
        T *d_buf;
        unsigned int *sample_state;     // 0 for no data, 1 for data
    protected:

    public:
        FilterbankBuffer(int id);
        FilterbankBuffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id);
        ~FilterbankBuffer(void);

        void Allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u);
        void deallocate(void);
        void SendToDisk(int idx, header_f head, std::string outdir);
        float **GetFilPointer(void) {return this->pd_filterbank;};
        ObsTime gettime(int index);
        int ready();
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void SendToRam(unsigned char *out, int idx, cudaStream_t &stream, int host_jump);
        void UpdateFilledTimes(ObsTime frame_time);
        void write(T *d_data, ObsTime frame_time, unsigned int amount, cudaStream_t stream);
        // add deleted copy, move, etc constructors
};

template<class T>
FilterbankBuffer<T>::FilterbankBuffer(int id) : gpuid(id)
{
    cudaSetDevice(gpuid);
    start = 0;
    end = 0;
}

template<class T>
FilterbankBuffer<T>::FilterbankBuffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id) : extra(extra_u),
                                                                                gulp(gulp_u),
                                                                                gulpno(gulpno_u),
                                                                                totsize(size_u),
                                                                                gpuid(id)
{
    start = 0;
    end = 0;
    //cudaMalloc((void**)&d_buf, totsize * sizeof(T));
    sample_state = new unsigned int[(int)totsize];
    std::fill(sample_state, sample_state + totsize, 0);
}

template<class T>
FilterbankBuffer<T>::~FilterbankBuffer()
{
    end = 0;
}

template<class T>
void FilterbankBuffer<T>::Allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u)
{
    fil_saved = 0;
    accumulate = acc_u;
    extra = extra_u;
    gulp = gulp_u;
    gulpno = gulpno_u;
    nchans = filchans;
    // size is the size of the buffer for the single Stokes parameter
    totsize = size_u;
    stokes = stokes_u;
    gulp_times = new ObsTime[gulpno];
    h_filterbank.resize(stokes);
    d_filterbank.resize(stokes);
    pd_filterbank = new float*[stokes];
    ph_filterbank = new float*[stokes];
    for (int ii = 0; ii < stokes; ii++) {
        // used to hold 2 full filterbank buffers
        h_filterbank[ii].resize((gulp + extra) * 2 * nchans);
        ph_filterbank[ii] = thrust::raw_pointer_cast(h_filterbank[ii].data());
        d_filterbank[ii].resize(totsize * nchans);
        pd_filterbank[ii] = thrust::raw_pointer_cast(d_filterbank[ii].data());
    }
    cudaCheckError(cudaMalloc((void**)&d_buf, totsize * stokes * sizeof(T)));
    sample_state = new unsigned int[(int)totsize];
    cudaCheckError(cudaHostAlloc((void**)&ph_fil, (gulp + extra) * nchans * stokes * 2 * sizeof(float), cudaHostAllocDefault));
    std::fill(sample_state, sample_state + totsize, 0);
}

template<class T>
void FilterbankBuffer<T>::deallocate(void)
{
    cudaCheckError(cudaFreeHost(ph_fil));
    cudaCheckError(cudaFree(d_buf));
    delete [] sample_state;
    delete [] gulp_times;
    delete [] pd_filterbank;
    delete [] ph_filterbank;
}

template<class T>
void FilterbankBuffer<T>::SendToDisk(int idx, header_f header, std::string outdir)
{
        SaveFilterbank(ph_fil, gulp + extra, (gulp + extra) * nchans * stokes * idx, header, stokes, fil_saved, outdir);
        fil_saved++;
        // need info from the telescope
}

template<class T>
int FilterbankBuffer<T>::ready()
{
    std::lock_guard<mutex> addguard(statemutex);
    // for now check only the last position for the gulp
    for (int ii = 0; ii < gulpno; ii++) {
        if (sample_state[(ii + 1) * gulp + extra - 1] == 1)
            return (ii + 1);
    }
    return 0;
}

template<class T>
void FilterbankBuffer<T>::GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs)
{
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulp + extra) * nchans * sizeof(float));
    for (int ii = 0; ii < stokes; ii++) {
        Transpose<<<1,nchans,0,stream>>>(pd_filterbank[ii] + (idx - 1) * gulp * nchans, d_transpose, nchans, gulp + extra);
        GetScaleFactors<<<1,nchans,0,stream>>>(d_transpose, d_means, d_rstdevs, nchans, gulp + extra, ii);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex.lock();
    sample_state[idx * gulp + extra - 1] = 0;
    statemutex.unlock();
}


template<class T>
void FilterbankBuffer<T>::SendToRam(unsigned char *out, int idx, cudaStream_t &stream, int host_jump)
{
    // which half of the RAM buffer we are saving into
    host_jump *= (gulp + extra) * nchans * stokes;
    // dump to the host memory only - not interested in the dedisperion in the dump mode
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump, pd_filterbank[0] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 1 * (gulp + extra) * nchans, pd_filterbank[1] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 2 * (gulp + extra) * nchans, pd_filterbank[2] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump + 3 * (gulp + extra) * nchans, pd_filterbank[3] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    statemutex.lock();
    // HACK: the call below is wrong - restarts the whole sample state
    //std::fill(sample_state, sample_state + totsize, 0);
    sample_state[idx * gulp + extra - 1] = 0;
    statemutex.unlock();
}


template<class T>
void FilterbankBuffer<T>::write(T *d_data, ObsTime frame_time, unsigned int amount, cudaStream_t stream)
{
    // need to make sure only one stream saves the data to the buffer
    // not really a problem anyway - only one DtD available at a time
    // we will save one data sample at a time, with fixed size
    // no need to check that there is enough space available to fit all the data before the end of the buffer
    std::lock_guard<mutex> addguard(buffermutex);
    int index = frame_time.framefromstart % totsize;
    if((index % gulp) == 0)
        gulp_times[index / gulp] = frame_time;
    if (end == totsize)    // reached the end of the buffer
        end = end - gulpno * gulp;    // go back to the start

    // TODO: try to come up with a slightly different implementation - DtD copies should be avoided whenever possible
    cudaCheckError(cudaMemcpyAsync(pd_filterbank[0] + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaCheckError(cudaMemcpyAsync(pd_filterbank[1] + index * amount, d_data + amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    //cudaCheckError(cudaMemcpyAsync(pd_filterbank[2] + index * amount, d_data + 2 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    //cudaCheckError(cudaMemcpyAsync(pd_filterbank[3] + index * amount, d_data + 3 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);
    sample_state[index] = 1;

    // need to save in two places in the buffer
    if (index >= gulpno * gulp) {
        // simplify the index algebra here
        // TODO: need to be actually sorted out properly
        cudaCheckError(cudaMemcpyAsync(d_buf + index - (gulpno * gulp) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        statemutex.lock();
        sample_state[index - (gulpno * gulp)] = 1;
        statemutex.unlock();
    }
    end = end + amount;
}

template<class T>
void FilterbankBuffer<T>::UpdateFilledTimes(ObsTime frame_time)
{
    std::lock_guard<mutex> addguard(statemutex);
    int framet = frame_time.framefromstart;
    int index = frame_time.framefromstart % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();

    for (int ii = 0; ii < accumulate; ii++) {
        index = framet % (gulpno * gulp);
        if((index % gulp) == 0)
            gulp_times[index / gulp] = frame_time;
        sample_state[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        if ((index < extra) && (framet > extra)) {
            sample_state[index + gulpno * gulp] = 1;
        }
        framet++;
    }
}

template<class T>
ObsTime FilterbankBuffer<T>::gettime(int index) {
    return gulp_times[index];
}

/*template<class T>
void FilterbankBuffer<T>::update(ObsTime frame_time)
{
    std::lock_guard<mutex> addguard(statemutex);
    int framet = frame_time.framet;
    int index = frame_time.framet % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    if((index % gulp) == 0)
        gulp_times[index / gulp] = frame_time;

    for (int ii = 0; ii < accumulate; ii++) {
        index = framet % (gulpno * gulp);
        sample_state[index] = 1;
        std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        std::cout.flush();
        if ((framet % totsize) >= (gulpno * gulp)) {
            sample_state[index + gulpno * gulp] = 1;
        }
        framet++;
    }
} */
#endif
