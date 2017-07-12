#ifndef _H_PAFRB_GPU_POOL
#define _H_PAFRB_GPU_POOL

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "config.hpp"
#include "dedisp/DedispPlan.hpp"
#include "filterbank_buffer.cuh"
#include "obs_time.hpp"

class GpuPool
{
    private:

        // NOTE: New, shiny names
        bool scaled_;
        bool verbose_;
        static bool working_;

        const unsigned int accumulate_;              //!< The number of 108us chunks to accumulate for the GPU processing
        const unsigned int avgfreq_;                 //!< Number of post-FFT frequency channels to average
        const unsigned int avgtime_;                 //!< Number of post-FFT time samples to average
        const unsigned int codiflen_;               //!< Length (in bytes) of the single CODIF data packet (just data, no header)
        const unsigned int fftbatchsize_;            //!< The number of 1MHz channels to process
        const unsigned int fftedsize_;              //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations        const unsigned int timescrunchedsize_;     //!< (size of single fft - 5) * # 1MHz channels
        const unsigned int fftpoints_;               //!< Size of the single FFT
        const unsigned int freqscrunchedsize_;     //!< timescrunchedsize_ / # frequency channels to average
        const unsigned int gpuid_;                      //!< Self-explanatory
        const unsigned int headlen_;                //!< Length (in bytes) of the CODIF header
        const unsigned int inbuffsize_;
        const unsigned int inchans_;                 //!< Number of 1MHz channels in the input data
        const unsigned int nopols_;                  //!< Number of polarisations in the input data
        const unsigned int noports_;
        const unsigned int nostokes_;                //!< Number of stoke parameters to generate and store
        const unsigned int nostreams_;               //!< Number of CUDA streams to use
        const unsigned int poolid_;
        const unsigned int rearrangedsize_;               //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int timescrunchedsize_;      //!< (size of single fft - 5) * # 1MHz channels

        cudaChannelFormatDesc arrangechandesc_;

        cudaStream_t dedispstream_;

        hd_pipeline singlepipeline_;
        hd_params singleparams_;

        InConfig config_;

        int fftsizes_[1];                   //<! Used to store GpuPool::fftpoint - cufftPlanMany() requirement

        ObsTime starttime_;

        std::string ipstring_;

        unsigned int beamno_;
        unsigned int dedispbuffersize_;
        unsigned int dedispdispersedsamples_;
        unsigned int dedispextrasamples_;
        unsigned int dedispgulpsamples_;
        unsigned int dedispnobuffers_;
        unsigned int filchans_;                //!< Number of output filterbank channels
        unsigned int gulpssent_;
        unsigned int packperbuffer_;
        unsigned int secondstorecord_;
        unsigned int usethreads_;

        std::vector<int> ports_;
        std::vector<std::thread> gputhreads_;
        std::vector<std::thread> receivethreads_;

        cufftComplex *dfftedbuffer_;            //!< Buffer for the signal after the FFT, powerscale() kernel input, holds GpuPool::fftedsize_ * GpuPool::nostreams_ elements
        cufftComplex *dstreambuffer_;         //!< Raw voltage device buffer, cufftExecC2C() input, holds GpuPool::rearrangedsize_ * GpuPool::nostreams_ elements

        cudaArray **arrange2darray_;
        cudaResourceDesc *arrangeresdesc_;
        cudaStream_t *gpustreams_;        //<! Pointer to the array of CUDA streams
        cudaTextureDesc *arrangetexdesc_;
        cudaTextureObject_t *arrangetexobj_;
        cufftHandle *fftplans_;           //<! Pointer to the array of cuFFT plans

        float **dfreqscrunchedbuffer_;
        float **dmeans_;
        float **drstdevs_;
        float **dtimescrunchedbuffer_;
        float **hfreqscrunchedbuffer_;
        float **hmeans_;
        float **hrstdevs_;
        float **htimescrunchedbuffer_;

        int *filedesc_;

        // TODO: This should really be a template
        std::unique_ptr<FilterbankBuffer<float>> filbuffer_;
        std::unique_ptr<DedispPlan> dedispplan_;

        unsigned char *hinbuffer_;           //!< Buffer for semi-arranged packets for the whole bandwidth and 128 time samples
        unsigned char *hstreambuffer_;       //!< Raw voltage host buffer, async copied to dstreambuffer_ in the GpuPool::worker()
        unsigned char **receivebuffers_;

        unsigned int *cudablocks_;       //<! Pointer to the array of block layouts for different kernels
        unsigned int *cudathreads_;      //<! Pointer to the array of thread layouts for different kernels

        // NOTE: Old, bad names

        bool *readybuffidx_;

        unsigned int filsize;
        // polarisations buffer
        int pol_begin;
        unsigned int *framenumbers_;               //!< Array for the absolute frame numbers for given buffers
        // GPU and thread stuff
        // raw voltage buffers
        // dstreambuffer_ is a cufftExecC2C() input
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
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

        void SendForDedispersion(void);

        void Initialise(void);

        static void HandleSignal(int signum);

        void FilterbankData(int stream);

        void ReceiveData(int portid, int recport);

};

#endif
