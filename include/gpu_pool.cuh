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

#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/DedispPlan.hpp"

class GpuPool
{
    private:

        // NOTE: New, shiny names
        bool scaled_;
        bool verbose_;
        static bool working_;

        const unsigned int accumulate_;              //!< The number of 108us chunks to accumulate for the GPU processing
        const unsigned int codiflen_;
        const unsigned int filchans_;                //!< Number of output filterbank channels
        const unsigned int headlen_;
        const unsigned int nopols_;                    //!< Number of polarisations in the input data
        const unsigned int nostokes_;                        //!< Number of stoke parameters to generate and store
        const unsigned int nostreams_;               //!< Number of CUDA streams to use
        const unsigned int poolid_;

        InConfig config_;

        unsigned int dedispextrasamples_;
        unsigned int dedispgulpsamples_;
        unsigned int havethreads_;
        // TODO: This should really be a template
        std::unique_ptr<Buffer<float>> filbuffer_;
        std::unique_ptr<DedispPlan> dedispplan_;

        std::vector<std::thread> gputhreads_;
        std::vector<std::thread> receivethreads_;
        // NOTE: Old, bad names

        //DedispPlan dedisp;
        hd_pipeline pipeline;
        hd_params params;
        int packcount;
        vector<thrust::device_vector<float>> dv_time_scrunch;   //!< Time scrunched buffer, addtime() kernel output, addchannel() kernel input; holds GpuPool::nostreams_ device vectors, each holding GpuPool::d_time_scrunch_size * GpuPool::stokes elements
        vector<thrust::device_vector<float>> dv_freq_scrunch;   //!< Frequency scrunched buffer, addchannel() kernel output; holds GpuPool::nostreams_ device vectors, each holding GpuPool::d_freq_scrunch_size * GpuPool::stokes elements

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
        std::mutex buffermutex;
        std::mutex printmutex;
        std::mutex workermutex;

        // SCALING
        float **h_means_;
        float **h_stdevs_;
        float **d_means_;
        float **d_rstdevs_;

        obs_time start_time;
        bool *bufidx_array;
        bool working;
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
        const unsigned int batchsize;               //!< The number of 1MHz channels to process
        const unsigned int fftpoint;                //!< Size of the single FFT
        const unsigned int timeavg;                 //!< Number of post-FFT time samples to average
        const unsigned int freqavg;                 //!< Number of post-FFT frequency channels to average
        unsigned int nchans;                        //!< Number of 1MHz channels in the input data
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
        cufftComplex *d_in = 0;         //!< Raw voltage device buffer, cufftExecC2C() input, holds GpuPool::d_in_size * GpuPool::nostreams_ elements
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
        cufftComplex *d_fft;            //!< Buffer for the signal after the FFT, powerscale() kernel input, holds GpuPool::d_fft_size * GpuPool::nostreams_ elements
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
        int beamno;
        size_t highest_buf;
        size_t highest_frame;
        cudaStream_t *mystreams;        //<! Pointer to the array of CUDA streams
        cufftHandle *myplans;           //<! Pointer to the array of cuFFT plans
        std::mutex datamutex;
        std::mutex workmutex;
        unsigned int *CUDAthreads;      //<! Pointer to the array of thread layouts for different kernels
        unsigned int *CUDAblocks;       //<! Pointer to the array of block layouts for different kernels

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
        GpuPool(int poolid, InConfig config);
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

        void SendForDedispersion(int dstream);

        void Initialise(void);

        static void HandleSignal(int signum);

        void FilterbankData(int stream);

        void ReceiveData(int ii);

};

#endif
