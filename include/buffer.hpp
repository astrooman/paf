#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

#include <algorithm>
#include <mutex>

using std::mutex;

struct obs_time {

    int start_epoch;            // refernece epoch at the start of the observation
    int start_second;           // seconds from the reference epoch at the start of the observation

}

template <class T>
class Buffer
{
    private:
        bool ready0, ready1, ready2;
        size_t size;            // total size of the data: #gulps * gulp size + extra samples for dedispersion
        size_t gulp;            // size of the single gulp
        size_t extra;           // number of extra time samples required to process the full gulp
        int gulpno;             // number of gulps required in the buffer
        mutex buffermutex;
        size_t start;
        size_t end;
        T *d_buf;
        unsigned int *sample_state;     // 0 for no data, 1 for data
    protected:

    public:
        Buffer();
        Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u);
        ~Buffer(void);


        void allocate(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u);
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
                                                                                size(size_u)
{
    start = 0;
    end = 0;
    cudaMalloc((void**)&d_buf, size * sizeof(T));
    sample_state = new unsigned int[(int)size];
    std::fill(sample_state, sample_state + size, 0);
}

template<class T>
Buffer<T>::~Buffer()
{
    end = 0;
    cudaFree(d_buf);
}

template<class T>
void Buffer<T>::allocate(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u)
{
    extra = extra_u;
    gulp = gulp_u;
    gulpno = gulpno_u;
    size = size_u;
    cudaMalloc((void**)&d_buf, size * sizeof(T));
    sample_state = new unsigned int[(int)size];
    std::fill(sample_state, sample_state + size, 0);
}

template<class T>
int Buffer<T>::ready()
{
    // for now check only the last position for the gulp
    for (int ii = 0; ii < gulpno; ii++) {
        if (sample_state[(ii + 1) * gulp + extra - 1] == 1)
            return (ii + 1);
    }
    return 0;
}

template<class T>
void Buffer<T>::send(unsigned char *out, int idx, cudaStream_t &stream)
{
    cudaMemcpyAsync(out, d_buf + (idx - 1) * gulp, (gulp + extra) * sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream);
    std::fill(sample_state, sample_state + size, 0);
}


template<class T>
void Buffer<T>::write(T *d_data, unsigned int index, unsigned int amount, cudaStream_t stream)
{
    // need to make sure only one stream saves the data to the buffer
    // we will save one data sample at a time, with fixed size
    // no need to check that there is enough space available to fit all the data before the end of the buffer
    std::lock_guard<mutex> addguard(buffermutex);
    index = index % size;
    if (end == size)    // reached the end of the buffer
        end = end - gulpno * gulp;    // go back to the start
    cudaMemcpyAsync(d_buf + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    sample_state[index] = 1;
    // need to save in two places in the buffer
    if (index >= gulpno * gulp) {
        // simplify the index algebra here
        cudaMemcpyAsync(d_buf + index - (gulpno * gulp) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        sample_state[index - (gulpno * gulp)] = 1;
    }
    end = end + amount;
}

#endif
