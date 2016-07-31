#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

/*! \file buffer.cuh
    \brief Defines the main buffer class.

    This is the buffer that is used to aggregate the FFTed data before it is sent to the dedispersion.
    Uses a slightly convoluted version of a ring buffer (the same data chunk is occasionally saved into two places).
*/

#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "errors.hpp"
#include "filterbank.hpp"

using std::mutex;
using std::vector;

struct obs_time {

    unsigned int start_epoch;            // reference epoch at the start of the observation
    unsigned int start_second;           // seconds from the reference epoch at the start of the observation
    unsigned int framet;                 // frame number from the start of the observation

};

template <class T>
class Buffer
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
        obs_time *gulp_times;
        T *d_buf;
        unsigned int *sample_state;     // 0 for no data, 1 for data
    protected:

    public:
        Buffer(int id);
        Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id);
        ~Buffer(void);

        void allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u);
        void dump(int idx, header_f head);
        float **get_pfil(void) {return this->pd_filterbank;};
        int ready();
        void send(unsigned char *out, int idx, cudaStream_t &stream, int host_jump);
        void update(obs_time frame_time);
        void write(T *d_data, obs_time frame_time, unsigned int amount, cudaStream_t stream);
        // add deleted copy, move, etc constructors
};

template<class T>
Buffer<T>::Buffer(int id) : gpuid(id)
{
    cudaSetDevice(gpuid);
    start = 0;
    end = 0;
}

template<class T>
Buffer<T>::Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id) : extra(extra_u),
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
Buffer<T>::~Buffer()
{
    end = 0;
    cudaCheckError(cudaFree(d_buf));
}

template<class T>
void Buffer<T>::allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u)
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
    gulp_times = new obs_time[gulpno];
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
    cudaCheckError(cudaHostAlloc((void**)&ph_fil, (gulp + extra) * 2 * nchans * sizeof(float), cudaHostAllocDefault));
    std::fill(sample_state, sample_state + totsize, 0);
}

/*template<class T>
void Buffer<T>::dump(int idx, header_f header)
{
        // idx will be use to tell which part of the buffer to dump
        std::cout << std::endl;
        save_filterbank(ph_filterbank, gulp + extra, (gulp + extra) * nchans * idx, header, stokes, fil_saved);
        fil_saved++;
        // need info from the telescope
}*/

template<class T>
void Buffer<T>::dump(int idx, header_f header)
{
        // idx will be use to tell which part of the buffer to dump
        std::cout << std::endl;
        save_filterbank2(ph_fil, gulp + extra, (gulp + extra) * nchans * idx, header, stokes, fil_saved);
        fil_saved++; 
        // need info from the telescope
}

template<class T>
int Buffer<T>::ready()
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
void Buffer<T>::send(unsigned char *out, int idx, cudaStream_t &stream, int host_jump)
{
    // which half of the RAM buffer we are saving into
    host_jump *= (gulp + extra) * nchans;
    // dump to the host memory only - not interested in the dedisperion in the dump mode
    //cudaCheckError(cudaMemcpyAsync(ph_filterbank[0] + host_jump, pd_filterbank[0] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil + host_jump, pd_filterbank[0] + (idx - 1) * gulp * nchans, (gulp + extra) * nchans * sizeof(T), cudaMemcpyDeviceToHost, stream));
    // that is a very quick hack to solve problems with hidden missing packets
    //cudaCheckError(cudaMemsetAsync(pd_filterbank[0] + (idx - 1) * gulp * nchans, 0, (gulp + extra) * nchans * sizeof(T), stream));
    //cudaCheckError(cudaMemcpyAsync(ph_filterbank[1] + host_jump, pd_filterbank[1] + (idx - 1) * gulp, (gulp + extra) * sizeof(T), cudaMemcpyDeviceToHost, stream));
    //cudaCheckError(cudaMemcpyAsync(ph_filterbank[2] + host_jump, pd_filterbank[2] + (idx - 1) * gulp, (gulp + extra) * sizeof(T), cudaMemcpyDeviceToHost, stream));
    //cudaCheckError(cudaMemcpyAsync(ph_filterbank[3] + host_jump, pd_filterbank[3] + (idx - 1) * gulp, (gulp + extra) * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    statemutex.lock();
    // HACK: the call below is wrong - restarts the whole sample state
    //std::fill(sample_state, sample_state + totsize, 0);
    sample_state[idx * gulp + extra - 1] = 0;
    statemutex.unlock();
}


template<class T>
void Buffer<T>::write(T *d_data, obs_time frame_time, unsigned int amount, cudaStream_t stream)
{
    // need to make sure only one stream saves the data to the buffer
    // not really a problem anyway - only one DtD available at a time
    // we will save one data sample at a time, with fixed size
    // no need to check that there is enough space available to fit all the data before the end of the buffer
    std::lock_guard<mutex> addguard(buffermutex);
    int index = frame_time.framet % totsize;
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
        // TODO: need to be actually sorte out properly
        cudaCheckError(cudaMemcpyAsync(d_buf + index - (gulpno * gulp) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        statemutex.lock();
        sample_state[index - (gulpno * gulp)] = 1;
        statemutex.unlock();
    }
    end = end + amount;
}

template<class T>
void Buffer<T>::update(obs_time frame_time)
{
    std::lock_guard<mutex> addguard(statemutex);
    int framet = frame_time.framet;
    int index = frame_time.framet % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int ii = 0; ii < accumulate; ii++) {
        index = framet % (gulpno * gulp);
        sample_state[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        if ((index < extra) && (framet > extra)) {
            sample_state[index + gulpno * gulp] = 1;
        }
        framet++;
    }
}


/*template<class T>
void Buffer<T>::update(obs_time frame_time)
{
    std::lock_guard<mutex> addguard(statemutex);
    int framet = frame_time.framet;
    int index = frame_time.framet % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
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
