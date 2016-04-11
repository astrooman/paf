#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <time.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

using std::cerr;
using std::cout;
using std::endl;

#define MEMALIGN 4096
void geterror(cufftResult res, std::string place);

struct my_stream {

	cudaStream_t stream;
	cufftHandle plan;
	int streamid;

};

class Streams {

	private:
		const unsigned int fftsize;
		const unsigned int batchsize;
		const unsigned int timesamp;

		bool *savbl;
		cudaStream_t *streams;
		cufftHandle *plans;
		int sizes[1];
	protected:

	public:
		// fs - size of single fft
		// bs - batchsize in one timesample
		// ts - number of timesamples
		Streams(unsigned int fs, unsigned int bs, unsigned int ts);
		~Streams(void);
		my_stream claim_stream(void);
		void free_stream(int streamid);
};

Streams::Streams(unsigned int fs, unsigned int bs, unsigned int ts) :
 					fftsize(fs), batchsize(bs), timesamp(ts) {

	savbl = new bool[4];
	streams = new cudaStream_t[4];
	plans = new cufftHandle[4];
	sizes[0] = fftsize;

	for (int ii = 0; ii < 4; ii++) {
		savbl[ii] = 1;		// make all streams available
		cudaStreamCreate(&streams[ii]);
		// NULL effectively switches the advanced data layout off
		cufftPlanMany(&plans[ii], 1, sizes, NULL, 1, fftsize, NULL, 1, fftsize, CUFFT_C2C, batchsize);
		cufftSetStream(plans[ii], streams[ii]);
	}

}

Streams::~Streams(void) {

	for (int ii = 0; ii < 4; ii++) {
		cufftDestroy(plans[ii]);
		cudaStreamDestroy(streams[ii]);
	}

	cout << "Destructor called";
	cout.flush();

	delete [] savbl;
	delete [] streams;
	delete [] plans;

}

// will be improved later
// now expects at least one stream to be free
my_stream Streams::claim_stream(void) {

	while(true) {
	for (int streamid = 0; streamid < 4; streamid++) {
		if (savbl[streamid] == 1) {
			//cout << "Stream " << streamid << " is free" << endl;
			savbl[streamid] = 0;
			return my_stream{streams[streamid], plans[streamid], streamid};
		}
	}
	}
}

void Streams::free_stream(int streamid) {
	savbl[streamid] = 1;
}



__global__ void poweradd(cufftComplex *in, float *out, unsigned int jump)
{

	int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int idx2 = blockIdx.x * blockDim.x + threadIdx.x + jump;

	if (idx1 < jump	) {
		float power1 = in[idx1].x * in[idx1].x + in[idx1].y * in[idx1].y;
		float power2 = in[idx2].x * in[idx2].x + in[idx2].y * in[idx2].y;
		out[idx1] = (power1 + power2) / 2;
	}

}

void tprocess(cufftComplex *h_in, cufftComplex *d_in, float *d_out, float* h_out, unsigned int size, cudaStream_t stream, cufftHandle plan, Streams &gtreams) {


}

void gpuprocess(cufftComplex *h_in, cufftComplex *d_in, float *d_out, float* h_out,
					unsigned int size, cudaStream_t stream, cufftHandle plan, Streams &gstreams, int id)
{

	unsigned int nthreads = 256;
	unsigned int nblocks = (size / 2 - 1) / nthreads + 1;

	//cout << "Stream: " << stream << endl;

	cudaMemcpyAsync(d_in, h_in, size * sizeof(cufftComplex), cudaMemcpyHostToDevice, stream);
	cufftExecC2C(plan, d_in, d_in, CUFFT_FORWARD);
	poweradd<<<nblocks, nthreads>>>(d_in, d_out, size / 2);
	cudaMemcpyAsync(h_out, d_out, size / 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	gstreams.free_stream(id);

}


int main(int argc, char* argv[])
{

	// this is proper test case with data flowing in
	// and multiple streams working on the data
    const unsigned int fftsize = 32;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
	const unsigned int timesamp = 2;		// the number fo timesamples we will store in buffer before processing
	const unsigned int totalsize = fftsize * batchsize * timesamp;
	// * 4 as we need memory for all 4 streams
    //const unsigned int alignin = (int)((totalsize * 4 * sizeof(cufftComplex) + MEMALIGN - 1) / MEMALIGN) * MEMALIGN;
	// returns half of the original samples after summing
	//const unsigned int alignout = (int)((totalsize * 4 / 2 * sizeof(float) + MEMALIGN - 1) / MEMALIGN) * MEMALIGN;

	std::thread thrd[4];

	timespec time1, time2;
	time1.tv_sec = 0;
	time1.tv_nsec = 104000;

	cufftComplex *h_in, *d_in;
	float *h_out, *d_out;

	cudaHostAlloc((void**)&h_in, totalsize * 4 * sizeof(cufftComplex), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_out, totalsize * 4 / 2 * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_in, totalsize * 4 * sizeof(cufftComplex));
	cudaMalloc((void**)&d_out, totalsize * 4 / 2 * sizeof(float));

	// need to initialise everything
    // data pointers, streams, etc
	Streams gstreams(fftsize, batchsize, timesamp);

	unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 arreng{seed};
	std::normal_distribution<float> arrdis(0.0, 1.0);

	for (int ii = 0; ii < totalsize * 4; ii++) {
            h_in[ii].x = arrdis(arreng);
            h_in[ii].y = arrdis(arreng);
    }

	// for now, we are just going to overwrite data over and over again
	for (unsigned int pack = 0; pack < 4; pack++) {

		// nanosleep will not sleep with sub-ms accuracy
		//nanosleep(&time1, &time2);
		for (int dumb = 0; dumb < 10000; dumb++) {}
		my_stream current = gstreams.claim_stream();
		unsigned int start = current.streamid * totalsize;
		thrd[current.streamid] = std::thread(tprocess, d_in + start, h_in + start, d_out + start, h_out + start, totalsize, current.stream, current.plan, gstreams); //, current.streamid);
		//cout << "Will get stream " << current.streamid;
		// need to check which stream is available
//gpuprocess(d_in + start, h_in + start, d_out + start, h_out + start, totalsize, current.stream, current.plan, gstreams, current.streamid);
		//cout << "Launch " << pack << " done" << endl;
		//cout.flush();

	}

	for (int ii = 0; ii < 4; ii++)
		thrd[ii].join();

	cudaFreeHost(h_in);
	cudaFreeHost(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

    cudaDeviceReset();
    return 0;

}
