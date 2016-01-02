/***********************************
VERSION OF THE FFT ALGORITHM FOR PAF
WITH FFT, POWER, AND AVERAGING ADDED

GENERAL TIMING RESULTS:
(E - EVENTS, P - NVPROF)


BOTH RESULTS QUOTED AS THERE ARE
SOME DISAGREEMENTS ON WHICH ONE IS
THE MOST RELIABLE ESTIMATE
***********************************/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

using std::cerr;
using std::cout;
using std::endl;
using std::string;


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

template <typename T>
void geterror(T res, std::string place)
{
    if ( (res != CUFFT_SUCCESS) && (res != cudaSuccess) )
        cout << "Error in " << place << "!! Error: " << res << endl;
}

void printhelp(void);

// GPU kernel
// need to figure out how to store the data efficienty
// don't want to introduce offset memory access
// which will significantly decrease the effective bandwidth

// of version with offset memory access
__global__ void poweraddkof(cufftComplex *arr_in, float *arr_out, unsigned int size)
{

	int index1 = blockIdx.x * blockDim.x + threadIdx.x;
	// offset introduced - can cause some slowing down
	int index2 = blockIdx.x * blockDim.x + threadIdx.x + size;

	if (index1 < size) {
		float power1 = arr_in[index1].x * arr_in[index1].x + arr_in[index1].y * arr_in[index1].y;
		float power2 = arr_in[index2].x * arr_in[index2].x + arr_in[index2].y * arr_in[index2].y;
		arr_out[index1] = (power1 + power2) / 2;
	}
}

int main(int argc, char* argv[])
{
	bool preinit = true;
	bool usekernel = true;
	string mode = "n";
    if (argc >= 2) {
		for (int ii = 0; ii < argc; ii++) {
			if (string(argv[ii]) == "-h") {
				printhelp();
			} else if (string(argv[ii]) == "-m") {
				ii++;
				mode = string(argv[ii]);
			} else if (string(argv[ii]) == "-p") {
				preinit = false;
			} else if (string(argv[ii]) == "-t") {
				usekernel = false;
			}
    		}
    }

	// that must be done before any CUDA context is created
	if (mode == "m")
		cudaSetDeviceFlags(cudaDeviceMapHost);

	if (preinit) {

		cout << "Pre-initialisation...\n";
    	PUSH_RANGE("FFT pre-init", 0)
    	// this should make the first proper FFT execution faster
    	cufftHandle preinit;
    	geterror(cufftPlan1d(&preinit, 32, CUFFT_C2C, 1), "init plan make");
    	POP_RANGE

	}

	// this stuff will stay the same between runs
	// const so I don't change it by mistake at some point
	const unsigned int arrsize = 32;
    const unsigned int fftsize = arrsize;
    const unsigned int batchsize = 1152;    // the number of FFTs we want to perform at once
	const unsigned int timesamp = 2;	// will need to process more than one timesamples for averaging
	const unsigned int fullsize = fftsize * batchsize * timesamp;
	const unsigned int memsize = fullsize * sizeof(cufftComplex);
	// limit is 1024 threads per block on all compute capablities
	// warp size is 32 on all compute capabilities
	unsigned int nthreads = 128;
	unsigned int nblocks = (fullsize / timesamp - 1) / nthreads + 1;
    // complex voltage goes in
	cufftComplex *h_inarray = new cufftComplex[fullsize];
	// time-averaged power goes out
	float *h_outarray = new float[fullsize / timesamp];
    int sizes[1] = {fftsize};
	unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 arreng{seed};
	std::normal_distribution<float> arrdis(0.0, 1.0);

	for (int ii = 0; ii < fullsize; ii++) {
			h_inarray[ii].x = arrdis(arreng);
			h_inarray[ii].y = arrdis(arreng);
	}

	if (mode == "n") {

		cout << "Will use standard memory copies...\n";

		cufftComplex *d_inarray;
		geterror(cudaMalloc((void**)&d_inarray, memsize), "device out malloc");
		// make sure memsize is even, i.e. timesamp is even
		// need only half of the original size for data averaged in time
		float *d_outarray;
		geterror(cudaMalloc((void**)&d_outarray, fullsize / timesamp * sizeof(float)), "device out malloc");

		PUSH_RANGE("Multi FFT init", 1)
		cufftHandle multiplan;
		geterror(cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "default plan make");
		POP_RANGE

		// time everything, together with memory copies
		PUSH_RANGE("Multi FFT exec", 2)
		geterror(cudaMemcpy(d_inarray, h_inarray, memsize, cudaMemcpyHostToDevice), "HtD copy");
		geterror(cufftExecC2C(multiplan, d_inarray, d_inarray, CUFFT_FORWARD), "default execution");
		poweraddkof<<<nblocks, nthreads>>>(d_inarray, d_outarray, fullsize / 2);
		geterror(cudaGetLastError(), "default kernel exec");
		geterror(cudaMemcpy(h_outarray, d_outarray, fullsize / timesamp * sizeof(float), cudaMemcpyDeviceToHost), "DtH copy");
		POP_RANGE

		geterror(cufftDestroy(multiplan), "default plan destroy");
		geterror(cudaFree(d_inarray), "device in free");
		geterror(cudaFree(d_outarray), "device out free");

	} else if (mode == "p") {

		cout << "Will use pinned memory";

		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);

		cufftComplex *h_inp, *d_inp;
		cufftComplex *h_inp2, *d_inp2;
		float *h_outp, *d_outp;
		float *h_outp2, *d_outp2;

		// make sure the size is a multiple of the page size
		// no need to actually use all of this memory in calculations
		// might not be necessary with out data as we end up with multiples of 4096
		// but better leave to be one the safe side
		int alignsizein = ((int)((int)memsize + MEMALIGN - 1) / MEMALIGN) * MEMALIGN;
		int alignsizeout = ((int)((int)(fullsize / 2) * sizeof(float) + MEMALIGN -1) / MEMALIGN) * MEMALIGN;

		cout << "Original in size: " << memsize << "B\n";
		cout << "Page multiple in size: " << alignsizein << "B\n";
		cout << "Page multiple out size: " << alignsizeout << "B\n";

		cufftHandle plan1, plan2;
		cufftPlanMany(&plan1, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp);
		cufftPlanMany(&plan2, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp);

		cufftSetStream(plan1, stream1);
		cufftSetStream(plan2, stream2);

		posix_memalign((void**)&h_inp, MEMALIGN, alignsizein);
		posix_memalign((void**)&h_inp2, MEMALIGN, alignsizein);
		posix_memalign((void**)&h_outp, MEMALIGN, alignsizeout);
		posix_memalign((void**)&h_outp2, MEMALIGN, alignsizeout);

		for (int ii = 0; ii < fullsize; ii++) {
			h_inp[ii].x = ii;
			h_inp[ii].y = 2 * ii;

			h_inp2[ii].x = ii;
			h_inp2[ii].y = 2 * ii;


                        //h_inp[ii].x = arrdis(arreng);
                        //h_inp[ii].y = arrdis(arreng);
        	}

		cudaHostRegister(h_inp, alignsizein, cudaHostRegisterDefault);
		cudaHostRegister(h_inp2, alignsizein, cudaHostRegisterDefault);
		cudaHostRegister(h_outp, alignsizeout, cudaHostRegisterDefault);
		cudaHostRegister(h_outp2, alignsizeout, cudaHostRegisterDefault);

		cudaHostGetDevicePointer((void**)&d_inp, (void*)h_inp, 0);
		cudaHostGetDevicePointer((void**)&d_outp, (void*)h_outp, 0);
		cudaHostGetDevicePointer((void**)&d_inp2, (void*)h_inp2, 0);
		cudaHostGetDevicePointer((void**)&d_outp2, (void*)h_outp2, 0);

		cufftExecC2C(plan1, d_inp, d_inp, CUFFT_FORWARD);
		cufftExecC2C(plan2, d_inp2, d_inp2, CUFFT_FORWARD);

		poweraddkof<<<nblocks, nthreads, 0, stream1>>>(d_inp, d_outp, fullsize / 2);
		poweraddkof<<<nblocks, nthreads, 0, stream2>>>(d_inp2, d_outp2, fullsize / 2);

		if(cudaGetLastError() != cudaSuccess)
			cout << "Error!!" << endl;

		cudaHostUnregister(h_inp);
		cudaHostUnregister(h_outp);
		cudaHostUnregister(h_inp2);
		cudaHostUnregister(h_outp2);

		free(h_inp);
		free(h_outp);
		free(h_inp2);
		free(h_outp2);

	} else if (mode == "m") {

		cout << "Will use mapped pinned memory...\n";

		cufftComplex *h_inarraym, *d_inarray;
		float *h_outarraym, *d_outarray;
		geterror(cudaHostAlloc((void**)&h_inarraym, memsize, cudaHostAllocMapped), "host in alloc");
		geterror(cudaHostAlloc((void**)&h_outarraym, fullsize / timesamp * sizeof(float), cudaHostAllocMapped), "host out alloc");
		geterror(cudaHostGetDevicePointer((void**)&d_inarray, (void*)h_inarraym, 0), "in dev pointer");
		geterror(cudaHostGetDevicePointer((void**)&d_outarray, (void*)h_outarraym, 0), "out dev pointer");

		for (int ii = 0; ii < fullsize; ii++) {
				h_inarraym[ii].x = arrdis(arreng);
				h_inarraym[ii].y = arrdis(arreng);
		}

		PUSH_RANGE("Multi mapped FFT init", 1)
		cufftHandle multiplan;
		geterror(cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "mapped plan make");
		POP_RANGE

		PUSH_RANGE("Multi mapped FFT exec", 2)
		geterror(cufftExecC2C(multiplan, d_inarray, d_inarray, CUFFT_FORWARD), "mapped execution");
		poweraddkof<<<nblocks, nthreads>>>(d_inarray, d_outarray, fullsize / 2);
		geterror(cudaGetLastError(), "mapped kernel exec");
		POP_RANGE

		geterror(cufftDestroy(multiplan), "mapped plan destroy");
		geterror(cudaFreeHost(h_inarraym), "host in free");
		geterror(cudaFreeHost(h_outarraym), "host out free");

		

	} else if (mode == "a") {

		cout << "Will use asynchronous memory copies...\n";

		// 2 streams should be enough, or might not show any benefit at all
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);

		cufftHandle multiplans1, multiplans2;
		geterror(cufftPlanMany(&multiplans1, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "async plan make 1");
		geterror(cufftPlanMany(&multiplans2, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize * timesamp), "async plan make 2");

		geterror(cufftSetStream(multiplans1, stream1), "FFT set stream 1");
		geterror(cufftSetStream(multiplans2, stream2), "FFT set stream 2");

		// must use pinned memory
		// very crude memory managment here
		// this will end up in loops at the end of the day
		// I will assume this stuff executes OK
		cufftComplex *h_inarraya1, *d_inarray1, *h_inarraya2, *d_inarray2;
		float *h_outarraya1, *d_outarray1, *h_outarraya2, *d_outarray2;
		cudaHostAlloc((void**)&h_inarraya1, memsize, cudaHostAllocDefault);
		cudaHostAlloc((void**)&h_inarraya2, memsize, cudaHostAllocDefault);

		for (int ii = 0; ii < fullsize; ii++) {
				h_inarraya1[ii].x = arrdis(arreng);
				h_inarraya1[ii].y = arrdis(arreng);
				h_inarraya2[ii].x = arrdis(arreng);
				h_inarraya2[ii].y = arrdis(arreng);
		}

		cudaHostAlloc((void**)&h_outarraya1, fullsize / timesamp * sizeof(float), cudaHostAllocDefault);
		cudaHostAlloc((void**)&h_outarraya2, fullsize / timesamp * sizeof(float), cudaHostAllocDefault);
		cudaMalloc((void**)&d_inarray1, memsize);
		cudaMalloc((void**)&d_inarray2, memsize);
		cudaMalloc((void**)&d_outarray1, fullsize / timesamp * sizeof(float));
		cudaMalloc((void**)&d_outarray2, fullsize / timesamp * sizeof(float));

		geterror(cudaMemcpyAsync(d_inarray1, h_inarraya1, memsize, cudaMemcpyHostToDevice, stream1), "HtD async copy 1");
		geterror(cudaMemcpyAsync(d_inarray2, h_inarraya2, memsize, cudaMemcpyHostToDevice, stream2), "HtD async copy 2");

		geterror(cufftExecC2C(multiplans1, d_inarray1, d_inarray1, CUFFT_FORWARD), "async execution 1");
		geterror(cufftExecC2C(multiplans2, d_inarray2, d_inarray2, CUFFT_FORWARD), "async execution 2");

		poweraddkof<<<nblocks, nthreads, 0, stream1>>>(d_inarray1, d_outarray1, fullsize / 2);
		poweraddkof<<<nblocks, nthreads, 0, stream2>>>(d_inarray2, d_outarray2, fullsize / 2);

		geterror(cudaMemcpyAsync(h_outarraya1, d_outarray1, fullsize / timesamp * sizeof(float), cudaMemcpyDeviceToHost, stream1), "DtH async copy 1");
		geterror(cudaMemcpyAsync(h_outarraya2, d_outarray2, fullsize / timesamp * sizeof(float), cudaMemcpyDeviceToHost, stream2), "DtH async copy 2");

		// I will assume this stuff executes OK as well
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);

		cudaFree(d_inarray1);
		cudaFree(d_inarray2);
		cudaFree(d_outarray1);
		cudaFree(d_outarray2);
		cudaFreeHost(h_inarraya1);
		cudaFreeHost(h_inarraya2);
		cudaFreeHost(h_outarraya1);
		cudaFreeHost(h_outarraya2);

	} else {
		cout << "Invalid memory mode option!! Will now quit!!";
	}

	geterror(cufftDestroy(preinit), "init plan destroy");
	delete [] h_inarray;
	delete [] h_outarray;

    cudaDeviceReset();

    return 0;

}

void printhelp(void)
{
	cout << "Test code for PAF FFT code" << endl << endl;
	cout << "Available options:" << endl;
	cout << "\t-p - switch pre-initialisation off" << endl;
	cout << "\t-t - use Thrust functions instead of custom kernels for power and averaging" << endl;
	cout << "\t-m - memory mode: n (default) - use cudaMemcpy()" << endl;
	cout << "\t\tp - use pinned memory, m - use mapped pinned memory, a - use asynchronous copies" << endl;

}
