#ifndef _H_PAFRB_GPU_POOL
#define _H_PAFRB_GPU_POOL

/*! \file pool_multi.cuh
    \brief Defines classes that are responsible for all the work done

*/

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/DedispPlan.hpp"

class GpuPool
{
    private:
        // that can be anything, depending on how many output bits we decide to use
        //Buffer<float> mainbuffer;
        std::unique_ptr<Buffer<float>> p_mainbuffer;
        //DedispPlan dedisp;
        std::unique_ptr<DedispPlan> p_dedisp;
        hd_pipeline pipeline;
        hd_params params;
        int packcount;
        vector<thrust::device_vector<float>> dv_time_scrunch;   //!< Time scrunched buffer, addtime() kernel output, addchannel() kernel input; holds GpuPool::nostreams device vectors, each holding GpuPool::d_time_scrunch_size * GpuPool::stokes elements
        vector<thrust::device_vector<float>> dv_freq_scrunch;   //!< Frequency scrunched buffer, addchannel() kernel output; holds GpuPool::nostreams device vectors, each holding GpuPool::d_freq_scrunch_size * GpuPool::stokes elements

        // networking
        unsigned char **rec_bufs;
        int *sfds;
        int filchansd4_;
/*
        float *pdv_power;
        float *pdv_time_scrunch;
        float *pdv_freq_scrunch;
*/
        // TEXTURING
        cudaChannelFormatDesc cdesc;
        cudaArray **d_array2Dp;
        cudaResourceDesc *rdesc;
        cudaTextureDesc *tdesc;
        cudaTextureObject_t *texObj;

        // MUTEXES
        mutex buffermutex;
        mutex printmutex;
        mutex workermutex;

        // SCALING
        bool scaled_;
        float **h_means_;
        float **h_stdevs_;
        float **d_means_;
        float **d_rstdevs_;

        obs_time start_time;
        config_s config_;
        bool *bufidx_array;
        bool working;
        bool verbose_;
        static bool working_;
        int record_;
        bool buffer_ready[2];
        bool worker_ready[2];
        int worker_frame[2];
        // const to be safe
        // keep d_in_size and d_fft_size separate just in case the way the fft is done changes
        const unsigned int d_rearrange_size;
        const unsigned int d_in_size;               //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_fft_size;              //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations        const unsigned int d_time_scrunch_size;     //!< (size of single fft - 5) * # 1MHz channels
        const unsigned int d_time_scrunch_size;      //!< (size of single fft - 5) * # 1MHz channels
        const unsigned int d_freq_scrunch_size;     //!< d_time_scrunch_size / # frequency channels to average
        const unsigned int accumulate;              //!< The number of 108us chunks to accumulate for the GPU processing
        const unsigned int batchsize;               //!< The number of 1MHz channels to process
        const unsigned int fftpoint;                //!< Size of the single FFT
        const unsigned int filchans_;                //!< Number of output filterbank channels
        const unsigned int nostreams;               //!< Number of CUDA streams to use
        const unsigned int timeavg;                 //!< Number of post-FFT time samples to average
        const unsigned int freqavg;                 //!< Number of post-FFT frequency channels to average
        const unsigned int npol;                    //!< Number of polarisations in the input data
        unsigned int nchans;                        //!< Number of 1MHz channels in the input data
        unsigned int stokes;                        //!< Number of stoke parameters to generate and store
        // one buffer
        unsigned int filsize;
        // polarisations buffer
        unsigned char *h_pol;           //!< Buffer for semi-arranged packets for the whole bandwidth and 128 time samples
        int gpuid;                      //!< Self-explanatory
        int pol_begin;
        int *frame_times;               //!< Array for the absolute frame numbers for given buffers
        // GPU and thread stuff
        // raw voltage buffers
        // d_in is a cufftExecC2C() input
        unsigned char *h_in = 0;        //!< Raw voltage host buffer, async copied to d_in in the GpuPool::worker()
        cufftComplex *d_in = 0;         //!< Raw voltage device buffer, cufftExecC2C() input, holds GpuPool::d_in_size * GpuPool::nostreams elements
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
        cufftComplex *d_fft;            //!< Buffer for the signal after the FFT, powerscale() kernel input, holds GpuPool::d_fft_size * GpuPool::nostreams elements
        // the detected signal buffer
        // powerscale() kernel output
        // addtime() kernel input
        float *d_power;                 //!< No longer in use
        // the time scrunched signal buffer
        // addtime() kernel output
        // addchannel() kernel input
        float *d_time_scrunch;          //!< No longer in use
        // the frequency schruned signal buffer
        // addchannel() kernel output
        float *d_freq_scrunch;          //!< No longer in use
        unsigned char *d_dedisp;        //!< Not in use in the dump mode
        unsigned char *d_search;        //!< Not in use in the dump mode

        unsigned int gulps_sent;
        unsigned int gulps_processed;
        int sizes[1];                   //<! Used to store GpuPool::fftpoint - cufftPlanMany() requirement
        int avt;
        int beamno;
        int poolid_;
        size_t highest_buf;
        size_t highest_frame;
        cudaStream_t *mystreams;        //<! Pointer to the array of CUDA streams
        cufftHandle *myplans;           //<! Pointer to the array of cuFFT plans
        mutex datamutex;
        mutex workmutex;
        unsigned int *CUDAthreads;      //<! Pointer to the array of thread layouts for different kernels
        unsigned int *CUDAblocks;       //<! Pointer to the array of block layouts for different kernels
        // containers
        // use queue as FIFO needed
        queue<pair<vector<cufftComplex>, obs_time>> mydata;
        //queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
        vector<thread> receive_threads;
        unsigned int dedisp_buffno;
        int pack_per_buf;
        size_t dedisp_buffsize;
        size_t dedisp_totsamples;
        std::string strip;
    protected:

    public:
        //! A default constructor.
        /*!
            Deleted
        */
        GpuPool(void) = delete;
        //! A constructor.
        /*!
            \param id the GPU id to be set using cudaSetDevice()
            \param config the configuration structure
        */
        GpuPool(int id, config_s config);
        ~GpuPool(void);
        //! A copy constructor.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GpuPool(const GpuPool &inpool) = delete;
        //! An assignment operator.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GpuPool& operator=(const GpuPool &inpool) = delete;
        //! Move constructor.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GpuPool(GpuPool &&inpool) = delete;
        //! Move assignment operator.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GpuPool& operator=(GpuPool &&inpool) = delete;

        //! Dedispersion thread worker
        /*! Responsible for picking up the data buffer when ready and dispatching it to the dedispersion (Buffer::send() method).
            In the filterbank dump mode, responsible for initialising the dump (Buffer::dump() method).
            \param dstream stream number, used to access stream from mystreams array
        */
        void SendForDedispersion(int dstream);
        //! Main GpuPool method.
        /*! Responsible for setting up the GPU execution.
            All memory allocated here, streams, cuFFT plans threads created here as well.
        */
        void Initialise(void);
        //! Handles the SIGINT signal. Must be static.
        /*!
            \param signum signal number - should be 2 for SIGINT
        */
        static void HandleSignal(int signum);
        //! Thread responsible for running the FFT.
        /*! There are 4 such threads per GPU - 4 streams per GPU used.
            Each thread is responsible for picking up the data from the queue (the thread yields if the is no data available), running the FFT and power, time scrunch and frequency scrunch kernels.
            After successfull kernel execution, writes to the main data buffer using write() Buffer method.
        */
        void FilterbankData(int stream);
        //! Calls async_receive_from() on the UDP socket.
        void ReceiveData(int ii);

};

#endif
