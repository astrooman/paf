#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

#include <algorithm>
#include <mutex>

#define SINGLE_GULP 131027

using std::mutex;

// make a template to support dfferent dedisp input types
template <class T>
class Buffer
{
    private:
        bool ready0, ready1, ready2;
        const size_t size;
        const int gulps;
        mutex buffermutex;
        size_t start;
        size_t end;
        T *d_buf;
        unsigned int *sample_state;
    protected:

    public:
        Buffer(size_t size);
        ~Buffer(void);

        int ready();
        void send();
        void write(T *d_data, unsigned int index, unsigned int amount, cudaStream_t stream);
        // add deleted copy, move, etc constructors
};

template<class T>
Buffer<T>::Buffer(size_t size) : size(size)
{
    start = 0;
    end = 0;
    cudaMalloc((void**)&d_buf, size * sizeof(T));
    sample_state = new int[size];
    std::fill(sample_state, sample_state + size, 0);
}

template<class T>
Buffer<T>::~Buffer()
{
    end = 0;
    cudaFree(d_buf);
}

template<class T>
int Buffer<T>::ready()
{
    if(sample_state[] == 1){
        return 1;
    } else if (sample_state[] == 1) {
        return 2;
    } else if (sample_state[] == 1){
        return 3;
    } else
        return 0;
}

template<class T>
// will return the pointer to the start of the data sent to dedispersion
void Buffer<T>::send()
{

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
        end = end - gulps * SINGLE_GULP;    // go back to the start
    cudaMemcpyAsync(d_buf + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    sample_state[index] = 1;
    // need to save in two places in the buffer
    if (index >= gulps * SINGLE_GULP) {
        cudaMemcpyAsync(d_buf + index - (gulps * SINGLE_GULP) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        sample_state[index - (gulps * SINGLE_GULP)] = 1;
    }
    end = end + amount;
}

#endif
