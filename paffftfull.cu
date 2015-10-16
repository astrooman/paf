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

void geterror(cufftResult res, std::string place);
void printhelp(void);

// GPU kernel
// need to figure out how to store the data efficienty
// don't want to introduce offset memory access
// which will significantly decrease the effective bandwidth

// st version with offset memory access
__global__ void poweraddkof(cufftComplex *arr_in, float *arr_out, unsigned int size)
{

	int index1 = blockIdx.x * blockDim.x + threadIdx.x;
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

	if (preinit) {

		cout << "Pre-initialisation...\n";
    	PUSH_RANGE("FFT pre-init", 0)
    	// this should make the first proper FFT execution faster
    	cufftHandle preinit;
    	cufftPlan1d(&preinit, 32, CUFFT_C2C, 1);
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

		// limit is 1024 threads per block on all compute capablities
		// warp size is 32 on all compute capabilities
		unsigned int nthreads = 256;
		unsigned int nblocks = (fullsize / timesamp - 1) / nthreads + 1;

		cout << "Will use standard memory copies...\n";

		cufftComplex *d_inarray;
		cudaMalloc((void**)&d_inarray, memsize);
		// make sure memsize is even, i.e. timesamp is even
		// need only half of the original size for data averaged in time
		float *d_outarray;
		cudaMalloc((void**)&d_outarray, fullsize / timesamp * sizeof(float));

		PUSH_RANGE("Multi FFT init", 1)
		cufftHandle multiplan;
		geterror(cufftPlanMany(&multiplan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize), "multi FFT plan");
		POP_RANGE

		// time everything, together with memory copies
		PUSH_RANGE("Multi FFT exec", 2)
		cudaMemcpy(d_inarray, h_inarray, memsize, cudaMemcpyHostToDevice);
		geterror(cufftExecC2C(multiplan, d_inarray, d_inarray, CUFFT_FORWARD), "multi FFT execution");
		poweraddkof<<<nblocks, nthreads>>>(d_inarray, d_outarray, fullsize / timesamp);
		cudaMemcpy(h_outarray, d_outarray, fullsize / timesamp * sizeof(float), cudaMemcpyDeviceToHost);
		POP_RANGE

		cufftDestroy(multiplan);
		cudaFree(d_inarray);
		cudaFree(d_outarray);

	} else if (mode == "p") {

		cout << "Will use pinned memory";

	} else if (mode == "m") {

		cout << "Will use mapped pinned memory...\n";

		cufftComplex d_inarray;
		cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaHostAlloc((void**)&h_inarray, memsize, cudaHostAllocMapped);
		cudaHostGetDevicePointer((void**)&d_inarray, (void*)h_inarray, 0);
		for (int ii = 0; ii < fullsize; ii++) {
				h_inarray[ii].x = arrdis(arreng);
				h_inarray[ii].y = arrdis(arreng);
		}

	} else if (mode == "a") {

		cout << "Will use asynchronous memory copies...\n";

	} else {
		cout << "Invalid memory mode option!! Will now quit!!";
	}

	cufftDestroy(preinit);
	delete [] h_inarray;
	delete [] h_outarray;

	/*

    cufftComplex *h_inarraym4;
    cufftComplex *d_inarraym4;

    cudaHostAlloc((void**)&h_inarraym4, sizeof(cufftComplex) * arrsize * batchsize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&d_inarraym4, (void *)h_inarraym4, 0);

    for (int ii = 0; ii < arrsize * batchsize; ii++) {
        h_inarraym4[ii].x = arrdis(arreng);
        h_inarraym4[ii].y = arrdis(arreng);

    }

    cudaEventRecord(init_start);
    cufftHandle multi4plan;
    geterror(cufftPlanMany(&multi4plan, 1, sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batchsize), "multi FFT 4 plan");
    cudaEventRecord(init_end);

    cudaEventRecord(exec_start);
    geterror(cufftExecC2C(multi4plan, d_inarraym4, d_inarraym4, CUFFT_FORWARD), "multi FFT 4 execution");
    cudaEventRecord(exec_end);

    cudaEventElapsedTime(&init_time, init_start, init_end);
    cudaEventElapsedTime(&exec_time, exec_start, exec_end);

    cout << "Init time: " << init_time << "ms\n";
    cout << "Exec time: " << exec_time << "ms\n";

    cudaFreeHost(h_inarraym4);
    cufftDestroy(multi4plan);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_end);
    cudaEventDestroy(exec_start);
    cudaEventDestroy(exec_end);

	*/

    cudaDeviceReset();

    return 0;

}

void geterror(cufftResult res, std::string place)
{
    if (res != CUFFT_SUCCESS)
        cout << "Error in " << place << "!! Error: " << res << endl;
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
