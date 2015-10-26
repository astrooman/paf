#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

using std::cerr;
using std::cout;
using std::endl;


static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0x00fd482f };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#define MEMALIGN 4096
void geterror(cufftResult res, std::string place);

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
		Streams(unsigned int fs, unsigned int bs, unsigned int ts;
		~Streams(void);

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

	delete [] streams;
	delete [] plans;

}

__global__ void poweradd(void)
{


}

void gpuprocess()
{

	cudaMemcpyAsync(d_in, h_in, memsize, cudaMemcpyHostToDevice, stream);
	cufftExecC2C(plan, d_in, d_in, CUFFT_FORWARD);
	poweradd<<<nblocks, nthreads>>>();

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

	cufftComplex *h_in, *d_in;
	float *h_out, *d_out;
    int sizes[1] = {fftsize};

	cudaHostAlloc((void**)&h_in, totalsize * 4 * sizeof(cufftComplex), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_out, totalsize * 4 / 2 * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_in, totalsize * 4 * sizeof(cufftComplex));
	cudaMalloc((void**)&d_out, totalsize * 4 / 2 * sizeof(float));

	// need to initialise everything
    // data pointers, streams, etc
	Streams gstreams(fftsize, batchsize, timesamp);

	for (unsigned int pack = 0; pack < 65536; pack++) {

		// need to check which stream is available
		gpuprocess();

	}

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 arreng{seed};
    std::normal_distribution<float> arrdis(0.0, 1.0);

    for (int ii = 0; ii < arrsize; ii++) {
            h_inarray[ii].x = arrdis(arreng);
            h_inarray[ii].y = arrdis(arreng);
    }

    cudaDeviceReset();

    return 0;

}

void geterror(cufftResult res, std::string place)
{
    if (res != CUFFT_SUCCESS)
        cout << "Error in " << place << "!! Error: " << res << endl;
}
