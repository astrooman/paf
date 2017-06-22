#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <pthread.h>

#include <fftw3.h>

#include "immintrin.h"

using std::cerr;
using std::cout;
using std::endl;
using std::mutex;
using std::thread;
using std::vector;

mutex coutmutex;
mutex planmutex;
void UnpackData(fftwf_complex *outi, fftwf_complex *outq, unsigned char* in, int tavg, int acc);
void PowerAverage(float *out, fftwf_complex *ini, fftwf_complex *inq, unsigned int timeavg, unsigned int freqavg, unsigned int acc);

void DoWork(int id) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id + 1, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cerr << "Error setting thread affinity for receive thread on port " << endl;
        exit(EXIT_FAILURE);
    }
    

    unsigned int accumulate = 8;
    unsigned int codiflen = 7168;
    unsigned int fftsize = 32;
    unsigned int nofpgas = 48;
    unsigned int nopols = 2;

    unsigned int repeat = 512;

    unsigned int insize = accumulate * codiflen * nofpgas;
    unsigned int unpackedsize = accumulate * 7 * 128 * nofpgas;

    unsigned int batchsize = unpackedsize / fftsize;
    unsigned char *codifbuffer = new unsigned char[insize];

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 codifeng{seed};
    std::uniform_int_distribution<> codifdist(0, 255);

    for (int isamp = 0; isamp < insize; isamp++)
        codifbuffer[isamp] = codifdist(codifeng);

    fftwf_complex *unpackedi, *unpackedq, *fftedi, *fftedq;
    fftwf_plan fftplani, fftplanq;

    unpackedi = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);
    unpackedq = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);
    fftedi = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);
    fftedq = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);
    
    int sizes[1] = {fftsize};

    // NOTE: Planning is not thread-safe
    planmutex.lock();
    fftplani = fftwf_plan_many_dft(1, sizes, batchsize, unpackedi, NULL, 1, fftsize, fftedi, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);
    fftplanq = fftwf_plan_many_dft(1, sizes, batchsize, unpackedq, NULL, 1, fftsize, fftedq, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);
    planmutex.unlock();

    auto runstart = std::chrono::high_resolution_clock::now();

    unsigned int timeavg = 1;
    unsigned int freqavg = 16;

    float *power = new float[unpackedsize];

    #pragma nounroll
    for (int irep = 0; irep < repeat; irep++) {
        UnpackData(unpackedi, unpackedq, codifbuffer, timeavg, accumulate);
        fftwf_execute(fftplani);
        fftwf_execute(fftplanq);
        PowerAverage(power, fftedi, fftedq, timeavg, freqavg, accumulate);
    }

    auto runend = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> rundur = runend - runstart;

    coutmutex.lock();
    cout << "On thread " << id << ":" << endl;
    cout << "Took " << rundur.count() << "s to run the unpacking and FFT" << endl;
    cout << "This is equal to " << rundur.count() * 1e+06 / (float)repeat << "us or " << rundur.count() * 1e+06 / (accumulate * 108.0) / (float)repeat << " times real time" << endl;
    coutmutex.unlock();

    fftwf_destroy_plan(fftplani);
    fftwf_destroy_plan(fftplanq);

    fftwf_free(fftedi);
    fftwf_free(fftedq);
    fftwf_free(unpackedi);
    fftwf_free(unpackedq);

    delete [] codifbuffer;


}

void UnpackData(fftwf_complex *outi, fftwf_complex *outq, unsigned char* in, int tavg = 1, int acc = 1) {

    int inidx = 0;
    int outidx = 0;

    for (int iacc = 0; iacc < acc; ++iacc) {
        for (int ifpga = 0; ifpga < 48; ++ifpga) {
            #pragma unroll(16)
            for (int itime = 0; itime < 128; ++itime) {
                for (int ichan = 0; ichan < 7; ++ichan) {
                    inidx = iacc * 7168 * 48 + (ifpga * 7 * 128 + itime * 7 + ichan) * 8;
                    //outidx = ifpga * 7 * 128 * acc + ichan * 128 * acc + 128 * iacc + itime;
                    outidx = (int)((iacc * 128 + itime) / (32 * tavg)) * 32 * tavg * 48 * 7 + (ifpga * 7 + ichan) * (32 * tavg) + (iacc * 128 + itime) % (32 * tavg);
                    outi[outidx][0] = static_cast<float>(static_cast<short>(in[inidx] | (in[inidx + 1] << 8)));
                    outi[outidx][1] = static_cast<float>(static_cast<short>(in[inidx + 2] | (in[inidx + 3] << 8)));
                    outq[outidx][0] = static_cast<float>(static_cast<short>(in[inidx + 4] | (in[inidx + 5] << 8)));
                    outq[outidx][1] = static_cast<float>(static_cast<short>(in[inidx + 6] | (in[inidx + 7] << 8)));
                    //__m128i _mm_set_epi16
                }
            }
        }
    }

}

void PowerAverage(float *out, fftwf_complex *ini, fftwf_complex *inq, unsigned int tavg, unsigned int favg, unsigned int acc) {

    unsigned int inidx = 0;
    unsigned int outidx = 0;

    float power = 0.0;
/*
    for (int iblock = 0; iblock < acc * 4 / tavg; iblock++) {
        for (int ichan = 0; ichan < 336; ichan++) {
            for (int ifft = 0; ifft < 32; ifft++) {
                for (int itavg = 0; itavg < tavg; itavg++) {
                    inidx = iblock * 336 * 32 * tavg + ichan * 32 * tavg + 32 * itavg + ifft;
                    __m128i power1 = _mm_set_epi32(ini[inidx][0], ini[inidx][0], ini[inidx][0], ini[inidx][0]);
                    __m128 power1f = _mm_cvtepi32_ps(power1);
                    power += ini[inidx][0] * ini[inidx][0] + ini[inidx][1] * ini[inidx][1] + inq[inidx][0] * inq[inidx][0] + inq[inidx][1] * inq[inidx][1];
                }
                outidx = iblock * 336 * 32 + ichan * 32 + ifft;
                out[outidx] = power;
                power = 0.0;
            }
        }

    }
*/
/*
    for (int iblock = 0; iblock < acc * 4 / tavg; iblock++) {
        for (int ichan = 0; ichan < 336; ichan++) {
            // NOTE: 32 * tavf will ALWAYS be divisible 
            for (int ifft = 0; ifft < 32 * tavg; ifft+=4) {
                inidx = iblock * 336 * 32 * tavg + ichan * 32 * tavg + ifft;                
                __m128i polii = _mm_set_epi32(ini[inidx][0], ini[inidx + 1][0], ini[inidx + 3][0], ini[inidx + 4][0]);
                __m128 polif = _mm_cvtepi32_ps(polii);
                __m128i polqi = _mm_set_epi32(inq[inidx][0], inq[inidx + 1][0], inq[inidx + 3][0], inq[inidx + 4][0]);
                __m128 polqf = _mm_cvtepi32_ps(polqi);
               
                __m128 powi = _mm_mul_ps(polif, polif);
                __m128 powq = _mm_mul_ps(polqf, polqf);
                //power = ini[inidx][0] * ini[inidx][0] + ini[inidx][1] * ini[inidx][1] + inq[inidx][0] * inq[inidx][0] + inq[inidx][1] * inq[inidx][1];
                outidx = iblock * 336 * 32 + ichan * 32 + ifft;
                //out[outidx] = power;
                power = 0.0;
            }
        }

    }
*/
/*
    for (int isamp = 0; isamp < acc * 336 * 128; isamp+=4) {
        inidx = isamp;
        __m128i polii = _mm_set_epi32(ini[inidx][0], ini[inidx + 1][0], ini[inidx + 3][0], ini[inidx + 4][0]);
        __m128 polif = _mm_cvtepi32_ps(polii);
        __m128i polqi = _mm_set_epi32(inq[inidx][0], inq[inidx + 1][0], inq[inidx + 3][0], inq[inidx + 4][0]);
        __m128 polqf = _mm_cvtepi32_ps(polqi);

        __m128 powi = _mm_mul_ps(polif, polif);
        __m128 powq = _mm_mul_ps(polqf, polqf);

    }
*/
    for (int isamp = 0; isamp < acc * 336 * 128; isamp+=8) {
        inidx = isamp;
        __m256i polii = _mm256_set_epi32(ini[inidx + 7][0], ini[inidx + 6][0], ini[inidx + 5][0], ini[inidx + 4][0], ini[inidx + 3][0], ini[inidx + 2][0], ini[inidx + 1][0], ini[inidx][0]);
        __m256 polif = _mm256_cvtepi32_ps(polii);
        __m256i polqi = _mm256_set_epi32(inq[inidx + 7][0], inq[inidx + 6][0], inq[inidx + 5][0], inq[inidx + 4][0], inq[inidx + 3][0], inq[inidx + 2][0], inq[inidx + 1][0], inq[inidx][0]);
        __m256 polqf = _mm256_cvtepi32_ps(polqi);

        __m256 powi = _mm256_mul_ps(polif, polif);
        __m256 powq = _mm256_mul_ps(polqf, polqf);

    }


}

int main(int argc, char *argv[])
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cerr << "Error setting thread affinity for receive thread on port " << endl;
        exit(EXIT_FAILURE);
    }


    vector<thread> workthreads;
    
    unsigned int nothreads = 6;

    for (int ithread = 0; ithread < nothreads; ithread++) {
        workthreads.push_back(thread(DoWork, ithread)); 
    }

    for (auto ithread = workthreads.begin(); ithread != workthreads.end(); ++ithread)
        ithread -> join();

    return 0;
}

