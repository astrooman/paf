#include <iostream>
#include <time.h>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cufft.h>

using std::cout;
using std::endl;

#define UINS 1000000	// microseconds in second
#define NINU 1000	// nanoseconds in microsecond
#define RUNS 25

int main(int argc, char* argv[])
{

	unsigned int fftsize = 32;
	unsigned int batchsize = 1152;
	unsigned int fullsize = 0;

	struct timespec astart, aend;
	struct timespec cstart, cend;
	std::vector<std::tuple<size_t, size_t, double>>  elapsed;


	cufftComplex *h_in = new cufftComplex[fftsize * batchsize * 512];

	cufftComplex mycomplex;
	mycomplex.x = 1.0;
	mycomplex.y = 1.0;

	std::fill(h_in, h_in + fftsize * batchsize * 512, mycomplex);

	for (unsigned int timesamp = 1; timesamp <= 512; timesamp *= 2)	{

		int *d_init;

		cudaMalloc((void**)&d_init, sizeof(int));
		cudaFree(d_init);

		fullsize = fftsize * batchsize * timesamp;

		cufftComplex *d_p[RUNS];

		clock_gettime(CLOCK_MONOTONIC, &astart);
		for (unsigned int run = 0; run < RUNS; run++){
			cudaMalloc((void**)&d_p[run], fullsize * sizeof(cufftComplex));
		}
		clock_gettime(CLOCK_MONOTONIC, &aend);

		clock_gettime(CLOCK_MONOTONIC, &cstart);
		for (unsigned int run = 0; run < RUNS; run++){
			cudaMemcpy(d_p[run], h_in, fullsize * sizeof(cufftComplex), cudaMemcpyHostToDevice);
		}
		clock_gettime(CLOCK_MONOTONIC, &cend);

		elapsed.push_back(std::make_tuple((aend.tv_sec - astart.tv_sec) * UINS / RUNS + (aend.tv_nsec - astart.tv_nsec) / NINU / RUNS,
							(cend.tv_sec - cstart.tv_sec) * UINS / RUNS + (cend.tv_nsec - cstart.tv_nsec) / NINU / RUNS,
							(double)(fullsize * sizeof(cufftComplex)) / (double)1024.0));

		for (unsigned int run = 0; run < RUNS; run++)
			cudaFree(d_p[run]);

		cudaDeviceReset();

	}

	for (std::vector<std::tuple<size_t, size_t, double>>::iterator ii = elapsed.begin(); ii != elapsed.end(); ii++)
		cout << std::get<2>(*ii) << "K took " << std::get<0>(*ii) << "us to allocate and " << std::get<1>(*ii) << "us to HtD copy \n";

	cudaDeviceReset();

	return 0;

}

