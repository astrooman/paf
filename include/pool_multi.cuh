#ifndef _H_PAFRB_POOL_MULTI
#define _H_PAFRB_POOL_MULTI

/*! \file pool_multi.cuh
    \brief Defines classes that are responsible for all the work done

*/

#include <memory>
#include <mutex>
#include <queue>
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

using boost::asio::ip::udp;
using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::vector;

class GPUpool;

/*! \class Oberpool
    \brief Main pool class, containter for GPUpool(s).

*/

class Oberpool
{
    private:

        int ngpus;

        std::vector<std::unique_ptr<GPUpool>> gpuvector;
        std::vector<std::thread> threadvector;
    protected:

    public:
        Oberpool(void) = delete;
        Oberpool(config_s config);
        Oberpool(const Oberpool &inpool) = delete;
        Oberpool& operator=(const Oberpool &inpool) = delete;
        Oberpool(Oberpool &&inpool) = delete;
        Oberpool& operator=(Oberpool &&inpool) = delete;
        ~Oberpool(void);

};

// TODO: clean this mess up!!

/*! \class GPUpool
    \brief Class responsible for managing the work on a single GPU.

*/

class GPUpool
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
        vector<thrust::device_vector<float>> dv_power;
        vector<thrust::device_vector<float>> dv_time_scrunch;
        vector<thrust::device_vector<float>> dv_freq_scrunch;

        udp::endpoint sender_endpoint;
	    vector<std::shared_ptr<udp::endpoint>> sender_endpoints;
        vector<udp::socket> sockets;
        boost::array<unsigned char, 7168 + 64> rec_buffer;

        std::shared_ptr<boost::asio::io_service> ios;
/*
        float *pdv_power;
        float *pdv_time_scrunch;
        float *pdv_freq_scrunch;
*/
        config_s _config;
        bool working;
        // const to be safe
        // keep d_in_size and d_fft_size separate just in case the way the fft is done changes
        const unsigned int d_in_size;                // size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_fft_size;              // size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_power_size;            // d_fft_size / # polarisations
        const unsigned int d_time_scrunch_size;     // (size of single fft - 5) * # 1MHz channels
        const unsigned int d_freq_scrunch_size;     // d_time_scrunch_size / # frequency channels to average
        const unsigned int batchsize;               // the number of FFTs to process at one
        const unsigned int fftpoint;                // size of the single FFT
        const unsigned int nostreams;               // # CUDA streams
        const unsigned int timeavg;                 // # time samples to average
        const unsigned int freqavg;                 // # frequency channels to average
        const unsigned int npol;                    // number of polarisations in the input data
        unsigned int nchans;
        unsigned int stokes;
        // one buffer
        unsigned int filsize;
        // polarisations buffer
        cufftComplex *h_pol;
        int gpuid;
        int pol_begin;
        // GPU and thread stuff
        // raw voltage buffers
        // d_in is a cufftExecC2C() input
        cufftComplex *h_in = 0;         //!< Raw voltage host buffer, async copied to d_in in the GPUpool::worker()
        cufftComplex *d_in = 0;         //!< Raw voltage device buffer, cufftExecC2C() input
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
        cufftComplex *d_fft;
        // the detected signal buffer
        // powerscale() kernel output
        // addtime() kernel input
        float *d_power;
        // the time scrunched signal buffer
        // addtime() kernel output
        // addchannel() kernel input
        float *d_time_scrunch;
        // the frequency schruned signal buffer
        // addchannel() kernel output
        float *d_freq_scrunch;          // the frequency scrunched signal buffer
        unsigned char *d_dedisp;        // dedispersion buffer - aggregated frequency scrunched buffer
        unsigned char *d_search;        // single pulse search buffer - dedispersion output

        unsigned int gulps_sent;
        unsigned int gulps_processed;
        int sizes[1];
        int avt;
        int highest_frame;
        cudaStream_t *mystreams;
        cufftHandle *myplans;
        mutex datamutex;
        mutex workmutex;
        unsigned int *CUDAthreads;
        unsigned int *CUDAblocks;
        // containers
        // use queue as FIFO needed
        queue<pair<vector<cufftComplex>, obs_time>> mydata;
        //queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
        unsigned int dedisp_buffno;
        int pack_per_buf;
        size_t dedisp_buffsize;
        size_t dedisp_totsamples;
    protected:

    public:
        //! A default constructor.
        /*!
            Deleted
        */
        GPUpool(void) = delete;
        //! A constructor.
        /*!
            \param id the GPU id to be set using cudaSetDevice()
            \param config the configuration structure
        */
        GPUpool(int id, config_s config);
        ~GPUpool(void);
        //! A copy constructor.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GPUpool(const GPUpool &inpool) = delete;
        //! An assignment operator.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GPUpool& operator=(const GPUpool &inpool) = delete;
        //! Move constructor.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GPUpool(GPUpool &&inpool) = delete;
        //! Move assignment operator.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GPUpool& operator=(GPUpool &&inpool) = delete;

        //! Add the data to the processing queue.
        /*! Called in GPupool::get_data() method.
            Adds a pair to the queue consistinf of the data buffer and associated time structure.
        */
        void add_data(cufftComplex *buffer, obs_time frame_time);
        //! Dedispersion thread worker
        /*! Responsible for picking up the data buffer when ready and dispatching it to the dedispersion (Buffer::send() method).
            In the filterbank dump mode, responsible for initialising the dump (Buffer::dump() method).
            \param dstream stream number, used to access stream from mystreams array
        */
        void dedisp_thread(int dstream);
        //! Main GPUpool method.
        /*! Responsible for setting up the GPU execution.
            All memory allocated here, streams, cuFFT plans threads created here as well.
        */
        void execute(void);
        //! Reads the data from the UDP packet.
        /*!
            \param *data buffer read by async_receive_from()
            \param fpga_id the FPGA number obtained from the sender's IP address; used to identify the frequency chunk and place in the buffer it will be saven in
            \param start_time structure containing the information when the current observation started (reference epoch and seconds from the reference epoch)
        */
        void get_data(unsigned char* data, int fpga_id, obs_time start_time);
        //! Thread responsible for running the FFT.
        /*! There are 4 such threads per GPU - 4 streams per GPU used.
            Each thread is responsible for picking up the data from the queue (the thread yields if the is no data available), running the FFT and power, time scrunch and frequency scrunch kernels.
            After successfull kernel execution, writes to the main data buffer using write() Buffer method.
        */
        void worker(int stream);
        //! Handler called from async_receive_from().
        /*! This function is responsible for handling the asynchronous receive on the socket.
            \param error error code
            \param bytes_transferred number of bytes received
            \param endpoint udp::endpoint object containing sender information (used to obtain the fpga_id from the sender's IP)

        */
        void receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint endpoint, int sockid);
        //! Calls async_receive_from() on the UDP socket.
        void receive_thread(int sockid);
        //! Single pulse search thread worker.
        /*! Responsible for picking up the dedispersed data buffer and dispatching it to the single pulse search pipeline.
            Calls hd_execute() and saves the filterbank if the appropriate single pulse has been detected.
            Disabled in the fulterbank dump mode.
        */
        void search_thread(int sstream);
};

#endif
