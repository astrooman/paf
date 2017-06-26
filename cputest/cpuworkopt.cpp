#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>

#include <fftw3.h>

#include "immintrin.h"

using std::cerr;
using std::cout;
using std::endl;
using std::mutex;
using std::string;
using std::thread;
using std::vector;

bool working = true;

mutex coutmutex;
mutex planmutex;
void UnpackData(fftwf_complex *outi, fftwf_complex *outq, unsigned char* in, int tavg, int acc);
void PowerAverage(float *out, fftwf_complex *ini, fftwf_complex *inq, unsigned int timeavg, unsigned int freqavg, unsigned int acc);

void BurnCpu(int id, int skipcores) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id + 1 + skipcores, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cerr << "Error setting thread affinity for burner thread on CPU " << id + 1 + skipcores << endl;
        exit(EXIT_FAILURE);
    }

    vector<float> myvalues;

    while(working) {
        
        unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 codifeng{seed};
        std::normal_distribution<float> codifdist(0.0, 131072.0);
        myvalues.push_back(codifdist(codifeng));
        if (myvalues.size() == 262144) {
            myvalues.clear();
        }
    }

}

void DoWork(int id) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(id + 1, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cerr << "Error setting thread affinity for work thread on CPU " << id + 1 << endl;
        exit(EXIT_FAILURE);
    }
    

    unsigned int accumulate = 1;
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
    
    int sizes[1] = {static_cast<int>(fftsize)};

    // NOTE: Planning is not thread-safe
    planmutex.lock();
    fftplani = fftwf_plan_many_dft(1, sizes, batchsize, unpackedi, NULL, 1, fftsize, fftedi, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);
    fftplanq = fftwf_plan_many_dft(1, sizes, batchsize, unpackedq, NULL, 1, fftsize, fftedq, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);
    planmutex.unlock();

    auto runstart = std::chrono::high_resolution_clock::now();

    unsigned int timeavg = 4;
    unsigned int freqavg = 16;

    unsigned int averagedsize = 27 * 336 / freqavg * 128 / fftsize / timeavg * accumulate;

    float *power = new float[averagedsize];
    //float *power = new float[unpackedsize];

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
                }
            }
        }
    }

}

void PowerAverage(float *out, fftwf_complex *ini, fftwf_complex *inq, unsigned int tavg, unsigned int favg, unsigned int acc) {

    unsigned int inidx = 0;
    unsigned int outidx = 0;

    float power = 0.0;
    
    __m256i summed = _mm256_set_epi32(0,0,0,0,0,0,0,0);
    __m256 summed1 = _mm256_cvtepi32_ps(summed);
    __m256 summed2 = _mm256_cvtepi32_ps(summed);
    __m256 summed3 = _mm256_cvtepi32_ps(summed);
    __m256 summed4 = _mm256_cvtepi32_ps(summed);

    // NOTE: Drop te last 5 channels from each 32-point FFT after the time averaging
    float taved[27 * 336];
    
    for (int ichan = 0; ichan < 336; ichan++) {
        for (int itavg = 0; itavg < tavg; ++itavg) {
            inidx = ichan * 32 * tavg + 32 * tavg;
/*
            __m256i polixi1 = _mm256_set_epi32(ini[inidx + 7][0], ini[inidx + 6][0], ini[inidx + 5][0], ini[inidx + 4][0], ini[inidx + 3][0], ini[inidx + 2][0], ini[inidx + 1][0], ini[inidx][0]);
            __m256i poliyi1 = _mm256_set_epi32(ini[inidx + 7][1], ini[inidx + 6][1], ini[inidx + 5][1], ini[inidx + 4][1], ini[inidx + 3][1], ini[inidx + 2][1], ini[inidx + 1][1], ini[inidx][1]);
            __m256 polixf1 = _mm256_cvtepi32_ps(polixi1);
            __m256 poliyf1 = _mm256_cvtepi32_ps(poliyi1);
            __m256i polqxi1 = _mm256_set_epi32(inq[inidx + 7][0], inq[inidx + 6][0], inq[inidx + 5][0], inq[inidx + 4][0], inq[inidx + 3][0], inq[inidx + 2][0], inq[inidx + 1][0], inq[inidx][0]);
            __m256i polqyi1 = _mm256_set_epi32(inq[inidx + 7][1], inq[inidx + 6][1], inq[inidx + 5][1], inq[inidx + 4][1], inq[inidx + 3][1], inq[inidx + 2][1], inq[inidx + 1][1], inq[inidx][1]);
            __m256 polqxf1 = _mm256_cvtepi32_ps(polqxi1);
            __m256 polqyf1 = _mm256_cvtepi32_ps(polqyi1);

            __m256 powix1 = _mm256_mul_ps(polixf1, polixf1);
            __m256 powiy1 = _mm256_mul_ps(poliyf1, poliyf1);
            __m256 powqx1 = _mm256_mul_ps(polqxf1, polqxf1);
            __m256 powqy1 = _mm256_mul_ps(polqyf1, polqyf1);

            __m256 powi1 = _mm256_add_ps(powix1, powiy1);
            __m256 powq1 = _mm256_add_ps(powqx1, powqy1);
            __m256 powtot1 = _mm256_add_ps(powi1, powq1);
            summed1 = _mm256_add_ps(summed1, powtot1);

            __m256i polixi2 = _mm256_set_epi32(ini[inidx + 15][0], ini[inidx + 14][0], ini[inidx + 13][0], ini[inidx + 12][0], ini[inidx + 11][0], ini[inidx + 10][0], ini[inidx + 9][0], ini[inidx + 8][0]);
            __m256i poliyi2 = _mm256_set_epi32(ini[inidx + 15][1], ini[inidx + 14][1], ini[inidx + 13][1], ini[inidx + 12][1], ini[inidx + 11][1], ini[inidx + 10][1], ini[inidx + 9][1], ini[inidx + 8][1]);
            __m256 polixf2 = _mm256_cvtepi32_ps(polixi2);
            __m256 poliyf2 = _mm256_cvtepi32_ps(poliyi2);
            __m256i polqxi2 = _mm256_set_epi32(inq[inidx + 15][0], inq[inidx + 14][0], inq[inidx + 13][0], inq[inidx + 12][0], inq[inidx + 11][0], inq[inidx + 10][0], inq[inidx + 9][0], inq[inidx + 8][0]);
            __m256i polqyi2 = _mm256_set_epi32(inq[inidx + 15][1], inq[inidx + 14][1], inq[inidx + 13][1], inq[inidx + 12][1], inq[inidx + 11][1], inq[inidx + 10][1], inq[inidx + 9][1], inq[inidx + 8][1]);
            __m256 polqxf2 = _mm256_cvtepi32_ps(polqxi2);
            __m256 polqyf2 = _mm256_cvtepi32_ps(polqyi2);

            __m256 powix2 = _mm256_mul_ps(polixf2, polixf2);
            __m256 powiy2 = _mm256_mul_ps(poliyf2, poliyf2);
            __m256 powqx2 = _mm256_mul_ps(polqxf2, polqxf2);
            __m256 powqy2 = _mm256_mul_ps(polqyf2, polqyf2);

            __m256 powi2 = _mm256_add_ps(powix2, powiy2);
            __m256 powq2 = _mm256_add_ps(powqx2, powqy2);
            __m256 powtot2 = _mm256_add_ps(powi2, powq2);
            summed2 = _mm256_add_ps(summed2, powtot2);

            __m256i polixi3 = _mm256_set_epi32(ini[inidx + 23][0], ini[inidx + 22][0], ini[inidx + 21][0], ini[inidx + 20][0], ini[inidx + 19][0], ini[inidx + 18][0], ini[inidx + 17][0], ini[inidx + 16][0]);
            __m256i poliyi3 = _mm256_set_epi32(ini[inidx + 23][1], ini[inidx + 22][1], ini[inidx + 21][1], ini[inidx + 20][1], ini[inidx + 19][1], ini[inidx + 18][1], ini[inidx + 17][1], ini[inidx + 16][1]);
            __m256 polixf3 = _mm256_cvtepi32_ps(polixi3);
            __m256 poliyf3 = _mm256_cvtepi32_ps(poliyi3);
            __m256i polqxi3 = _mm256_set_epi32(inq[inidx + 23][0], inq[inidx + 22][0], inq[inidx + 21][0], inq[inidx + 20][0], inq[inidx + 19][0], inq[inidx + 18][0], inq[inidx + 17][0], inq[inidx + 16][0]);
            __m256i polqyi3 = _mm256_set_epi32(inq[inidx + 23][1], inq[inidx + 22][1], inq[inidx + 21][1], inq[inidx + 20][1], inq[inidx + 19][1], inq[inidx + 18][1], inq[inidx + 17][1], inq[inidx + 16][1]);
            __m256 polqxf3 = _mm256_cvtepi32_ps(polqxi3);
            __m256 polqyf3 = _mm256_cvtepi32_ps(polqyi3);

            __m256 powix3 = _mm256_mul_ps(polixf3, polixf3);
            __m256 powiy3 = _mm256_mul_ps(poliyf3, poliyf3);
            __m256 powqx3 = _mm256_mul_ps(polqxf3, polqxf3);
            __m256 powqy3 = _mm256_mul_ps(polqyf3, polqyf3);

            __m256 powi3 = _mm256_add_ps(powix3, powiy3);
            __m256 powq3 = _mm256_add_ps(powqx3, powqy3);
            __m256 powtot3 = _mm256_add_ps(powi3, powq3);
            summed3 = _mm256_add_ps(summed3, powtot3);

            __m256i polixi4 = _mm256_set_epi32(ini[inidx + 31][0], ini[inidx + 30][0], ini[inidx + 29][0], ini[inidx + 28][0], ini[inidx + 27][0], ini[inidx + 26][0], ini[inidx + 25][0], ini[inidx + 24][0]);
            __m256i poliyi4 = _mm256_set_epi32(ini[inidx + 31][1], ini[inidx + 30][1], ini[inidx + 29][1], ini[inidx + 28][1], ini[inidx + 27][1], ini[inidx + 26][1], ini[inidx + 25][1], ini[inidx + 24][1]);
            __m256 polixf4 = _mm256_cvtepi32_ps(polixi4);
            __m256 poliyf4 = _mm256_cvtepi32_ps(poliyi4);
            __m256i polqxi4 = _mm256_set_epi32(inq[inidx + 31][0], inq[inidx + 30][0], inq[inidx + 29][0], inq[inidx + 28][0], inq[inidx + 27][0], inq[inidx + 26][0], inq[inidx + 25][0], inq[inidx + 24][0]);
            __m256i polqyi4 = _mm256_set_epi32(inq[inidx + 31][1], inq[inidx + 30][1], inq[inidx + 29][1], inq[inidx + 28][1], inq[inidx + 27][1], inq[inidx + 26][1], inq[inidx + 25][1], inq[inidx + 24][1]);
            __m256 polqxf4 = _mm256_cvtepi32_ps(polqxi4);
            __m256 polqyf4 = _mm256_cvtepi32_ps(polqyi4);

            __m256 powix4 = _mm256_mul_ps(polixf4, polixf4);
            __m256 powiy4 = _mm256_mul_ps(poliyf4, poliyf4);
            __m256 powqx4 = _mm256_mul_ps(polqxf4, polqxf4);
            __m256 powqy4 = _mm256_mul_ps(polqyf4, polqyf4);

            __m256 powi4 = _mm256_add_ps(powix4, powiy4);
            __m256 powq4 = _mm256_add_ps(powqx4, powqy4);
            __m256 powtot4 = _mm256_add_ps(powi4, powq4);

            summed4 = _mm256_add_ps(summed4, powtot4);
*/
            // NOTE: Loading whole complex numbers does not decrease the total number of loads, but there will be no weird conversions between ints and floats and it is long
            // NOTE: Each load now contains 4 full complex numbers (instead of 8 halves as before)
            __m256 poli1 = _mm256_loadu_ps(reinterpret_cast<float*>(ini));
            __m256 polq1 = _mm256_loadu_ps(reinterpret_cast<float*>(inq));
            
            __m256 sqri1 = _mm256_mul_ps(poli1, poli1);
            __m256 sqrq1 = _mm256_mul_ps(polq1, polq1);
            __m256 part1 = _mm256_add_ps(sqri1, sqrq1);

            __m256 poli2 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 4));
            __m256 polq2 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 4));

            __m256 sqri2 = _mm256_mul_ps(poli2, poli2);
            __m256 sqrq2 = _mm256_mul_ps(polq2, polq2);
           __m256 part2 = _mm256_add_ps(sqri2, sqrq2);
           
           __m256 comb12 = _mm256_hadd_ps(part1, part2);
           summed1 = _mm256_add_ps(summed1, comb12);


            __m256 poli3 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 8));
            __m256 polq3 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 8));

            __m256 sqri3 = _mm256_mul_ps(poli3, poli3);
            __m256 sqrq3 = _mm256_mul_ps(polq3, polq3);

            __m256 part3 = _mm256_add_ps(sqri3, sqrq3);


            __m256 poli4 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 12));
            __m256 polq4 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 12));

            __m256 sqri4 = _mm256_mul_ps(poli4, poli4);
            __m256 sqrq4 = _mm256_mul_ps(polq4, polq4);

           __m256 part4 = _mm256_add_ps(sqri4, sqrq4);
           __m256 comb34 = _mm256_hadd_ps(part3, part4);
           summed2 = _mm256_add_ps(summed2, comb34);

            __m256 poli5 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 16));
            __m256 polq5 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 16));

            __m256 sqri5 = _mm256_mul_ps(poli5, poli5);
            __m256 sqrq5 = _mm256_mul_ps(polq5, polq5);
            __m256 part5 = _mm256_add_ps(sqri5, sqrq5);

            __m256 poli6 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 20));
            __m256 polq6 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 20));

            __m256 sqri6 = _mm256_mul_ps(poli6, poli6);
            __m256 sqrq6 = _mm256_mul_ps(polq6, polq6);
           __m256 part6 = _mm256_add_ps(sqri6, sqrq6);

           __m256 comb56 = _mm256_hadd_ps(part5, part6);
           summed3 = _mm256_add_ps(summed3, comb56);              


            __m256 poli7 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 24));
            __m256 polq7 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 24));

            __m256 sqri7 = _mm256_mul_ps(poli7, poli7);
            __m256 sqrq7 = _mm256_mul_ps(polq7, polq7);
            __m256 part7 = _mm256_add_ps(sqri7, sqrq7);

            __m256 poli8 = _mm256_loadu_ps(reinterpret_cast<float*>(ini + 28));
            __m256 polq8 = _mm256_loadu_ps(reinterpret_cast<float*>(inq + 28));

            __m256 sqri8 = _mm256_mul_ps(poli8, poli8);
            __m256 sqrq8 = _mm256_mul_ps(polq8, polq8);
           __m256 part8 = _mm256_add_ps(sqri8, sqrq8);

           __m256 comb78 = _mm256_hadd_ps(part7, part8);
           summed4 = _mm256_add_ps(summed4, comb78);


        }

        outidx = 27 * ichan;

        float *floatsum1 = (float*)&summed1;
        float *floatsum2 = (float*)&summed2;
        float *floatsum3 = (float*)&summed3;
        #pragma unroll
        for (int isum = 0; isum < 8; isum++) {
           taved[outidx + isum] = floatsum1[isum];
           taved[outidx + 8 + isum] = floatsum2[isum];
           taved[outidx + 16 + isum] = floatsum3[isum];
        }

        float *floatsum4 = (float*)&summed4;
        taved[outidx + 24] = floatsum4[0];
        taved[outidx + 25] = floatsum4[1];
        taved[outidx + 26] = floatsum4[2];
    }

    float faved = 0.0;

    // NOTE: 16 is really the highest frequency average we can get
    for (int ifreq = 0; ifreq < 336 * 27 / favg; ++ifreq) {
        for (int ifavg; ifavg < favg; ++ifavg) {
            inidx =   ifreq * favg + ifavg;
            faved += taved[inidx];
        }
        outidx = ifreq;
        out[outidx] = faved;
        faved = 0.0; 
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

    bool burners = false;
    
    for (int iarg = 0; iarg < argc; ++iarg) {
        if (string(argv[iarg]) == "-b") {
            burners = true;
        }
    }

    vector<thread> workthreads;
    vector<thread> burnerthreads;
    
    unsigned int nothreads = 6;
    unsigned int totthreads = 10;

    for (int ithread = 0; ithread < nothreads; ithread++) {
        workthreads.push_back(thread(DoWork, ithread)); 
    }

    if (burners) {
        for (int ithread = 0; ithread < totthreads - nothreads - 1; ithread++) {
            burnerthreads.push_back(thread(BurnCpu, ithread, nothreads));
        }
    }

    for (auto ithread = workthreads.begin(); ithread != workthreads.end(); ++ithread)
        ithread -> join();

    working = false;

    if (burners) {
        for (auto ithread = burnerthreads.begin(); ithread != burnerthreads.end(); ++ithread)
            ithread -> join();
    }

    return 0;
}

