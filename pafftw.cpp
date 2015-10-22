#include <chrono>
#include <fftw3.h>
#include <iostream>
#include <random>
#include <time.h>

using std::cout;
using std::endl;

#define UINS 1000000
#define NINS 1000
#define RUNS 50


void poweradd(fftw_complex *in, float* out, unsigned int size)
{

	for (int ii = 0; ii < size / 2; ii++)
	{
		out[ii] = (in[ii][0] * in[ii][0] + in[ii][1] * in[ii][1]) + (in[ii+size/2][0] * in[ii+size/2][0] + in[ii+size/2][1] * in[ii+size/2][1]);
	}
}

int main(int argc, char* argv[])
{

	size_t elapsed;
	struct timespec start, end;
	unsigned int fftsize = 32;
	unsigned int batchsize = 1152;
	unsigned int timesamp = 2;
	unsigned int fullsize = fftsize * batchsize * timesamp;
	int sizes[1] = {(int)fftsize};

	fftw_complex *inarray; //, *outarray;

	inarray = (fftw_complex*)fftw_malloc(fullsize * sizeof(fftw_complex));
	//outarray = (fftw_complex*)fftw_malloc(fullsize * sizeof(fftw_complex));

	float *outarray = new float[fullsize / 2];

	unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 arreng{seed};
	std::normal_distribution<float> arrdis(0.0, 1.0);

	for (int ii = 0; ii < fullsize; ii++) {
			inarray[ii][0] = arrdis(arreng);
			inarray[ii][1] = arrdis(arreng);
	}

	fftw_plan plan = fftw_plan_many_dft(1, sizes, batchsize * timesamp, inarray, NULL, 1, fftsize, inarray, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int run = 0; run < RUNS; run++) {
		fftw_execute(plan);
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = (end.tv_sec - start.tv_sec) * UINS / RUNS + (end.tv_nsec - start.tv_nsec) / NINS / RUNS;
	cout << "FFT run took: " << elapsed << "\xC2\xB5s\n";

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int run = 0; run < RUNS; run++) {
		poweradd(inarray, outarray, fullsize);
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = (end.tv_sec - start.tv_sec) * UINS / RUNS + (end.tv_nsec - start.tv_nsec) / NINS / RUNS;
	cout << "Power/add run took: " << elapsed << "\xC2\xB5s\n";


	fftw_destroy_plan(plan);
	fftw_free(inarray);

	return 0;

}
