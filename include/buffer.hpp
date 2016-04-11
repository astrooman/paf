#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

#include <algorithm>
#include <mutex>
#include <vector>

#include <filterbank.hpp>
#include <thrust/host_vector.h>
#include <thrsust/device_vector.h>

using std::mutex;

struct obs_time {

    int start_epoch;            // refernece epoch at the start of the observation
    int start_second;           // seconds from the reference epoch at the start of the observation
    int framet;                 // frame number from the start of the observation

}

template <class T>
class Buffer
{
    private:
        vector<thrust::device_vector<float>> d_filterbank;              // stores different Stoke parameters
        vector<thrust::host_vector<float>> h_filterbank;                // stores Stokes parameters in the RAM buffer
        float **pd_filterbank;                                          // array of raw pointers to Stoke parameters device vectors
        float **ph_filterbank;                                          // same as above but for host vector
        size_t totsize;            // total size of the data: #gulps * gulp size + extra samples for dedispersion
        size_t gulp;            // size of the single gulp
        size_t extra;           // number of extra time samples required to process the full gulp
        int gulpno;             // number of gulps required in the buffer
        int stokes;             // number of Stokes parameters to keep in the buffer
        mutex buffermutex;
        mutex statemutex;
        size_t start;
        size_t end;
        obs_time *gulp_times;
        T *d_buf;
        unsigned int *sample_state;     // 0 for no data, 1 for data
    protected:

    public:
        Buffer();
        Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u);
        ~Buffer(void);

        void allocate(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u);
        void dump();
        int ready();
        void send(unsigned char *out, int idx, cudaStream_t &stream);
        void write(T *d_data, unsigned int index, unsigned int amount, cudaStream_t stream);
        // add deleted copy, move, etc constructors
};

template<class T>
Buffer<T>::Buffer()
{
    start = 0;
    end = 0;
}

template<class T>
Buffer<T>::Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u) : extra(extra_u),
                                                                                gulp(gulp_u),
                                                                                gulpno(gulpno_u),
                                                                                totsize(size_u)
{
    start = 0;
    end = 0;
    cudaMalloc((void**)&d_buf, totsize * sizeof(T));
    sample_state = new unsigned int[(int)totsize];
    std::fill(sample_state, sample_state + totsize, 0);
}

template<class T>
Buffer<T>::~Buffer()
{
    end = 0;
    cudaFree(d_buf);
}

template<class T>
void Buffer<T>::allocate(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int stokes_u)
{
    extra = extra_u;
    gulp = gulp_u;
    gulpno = gulpno_u;
    // size is the size of the buffer for the single Stokes parameter
    totsize = size_u;
    stokes = stokes_u;
    gulp_times = new obs_time[gulpno];
    filterbank.resize(stokes);
    pd_filterbank = new float*[stokes];
    ph_filterbank = new float*[stokes];
    for (int ii = 0; ii < stokes; ii++) {
        h_filterbank[ii].resize((gulp + extra) * 2);
        ph_filterbank[ii] = thrust::raw_pointer_cast(h_filterbank[ii].data());
        d_filterbank[ii].resize(totsize);
        pd_filterbank[ii] = thrust::raw_pointer_cast(d_filterbank[ii].data());
    }
    cudaMalloc((void**)&d_buf, totsize * stokes * sizeof(T));
    sample_state = new unsigned int[(int)totsize];
    std::fill(sample_state, sample_state + totsize, 0);
}

template<class T>
void Buffer<T>::dump()
{
        // idx will be use to tell which
        save_filterbank(ph_filterbank, gulp + extra, gulp * idx);
        // need info from the telescope
}

template<class T>
int Buffer<T>::ready()
{
    std::lock_quard<mutex> addguard(statemutex);
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
    host_jump *= (gulp + extra);

    cudaMemcpyAsync(out, d_buf + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(ph_filterbank[0] + host_jump, pd_filterbank[0] + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ph_filterbank[1] + host_jump, pd_filterbank[1] + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ph_filterbank[2] + host_jump, pd_filterbank[2] + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ph_filterbank[3] + host_jump, pd_filterbank[3] + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);

    statemutex.lock();
    std::fill(sample_state, sample_state + totsize, 0);
    statemutex.unlock();
}


template<class T>
void Buffer<T>::write(T *d_data, obs_time frame_time, unsigned int amount, cudaStream_t stream)
{
    // need to make sure only one stream saves the data to the buffer
    // we will save one data sample at a time, with fixed size
    // no need to check that there is enough space available to fit all the data before the end of the buffer
    std::lock_guard<mutex> addguard(buffermutex);
    int index = frame_time.framet % totsize;
    if((index % gulp) = 0)
        gulp_times[index / gulp] = frame_time;
    if (end == totsize)    // reached the end of the buffer
        end = end - gulpno * gulp;    // go back to the start

    cudaMemcpyAsync(pd_filterbank[0] + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pd_filterbank[1] + index * amount, d_data + amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pd_filterbank[2] + index * amount, d_data + 2 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pd_filterbank[3] + index * amount, d_data + 3 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    //cudaMemcpyAsync(d_buf + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    sample_state[index] = 1;
    // need to save in two places in the buffer
    if (index >= gulpno * gulp) {
        // simplify the index algebra here
        cudaMemcpyAsync(d_buf + index - (gulpno * gulp) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        statemutex.lock();
        sample_state[index - (gulpno * gulp)] = 1;
        statemutex.unlock();
    }
    end = end + amount;
}

#endif
