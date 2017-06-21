#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#include <fftw3.h>

using std::cerr;
using std::cout;
using std::endl;

void UnpackData(fftwf_complex *out, unsigned char* in, int acc = 1) {

    int inidx = 0;
    int outidx = 0;

    for (int iacc = 0; iacc < acc; ++iacc) {
        for (int ifpga = 0; ifpga < 48; ++ifpga) {
            for (int itime = 0; itime < 128; ++itime) {
                #pragma unroll(7)
                for (int ichan = 0; ichan < 7; ++ichan) {
                    inidx = iacc * 7168 * 48 + (ifpga * 7 * 128 + itime * 7 + ichan) * 8;
                    outidx = ifpga * 7 * 128 * acc + ichan * 128 * acc + 128 * iacc + itime;
                    out[outidx][0] = static_cast<float>(static_cast<short>(in[inidx] | (in[inidx + 1] << 8)));
                    out[outidx][1] = static_cast<float>(static_cast<short>(in[inidx + 2] | (in[inidx + 3] << 8)));
                    out[outidx + 48 * 7 * 128][0] = static_cast<float>(static_cast<short>(in[inidx + 4] | (in[inidx + 5] << 8)));
                    out[outidx + 48 * 7 * 128][1] = static_cast<float>(static_cast<short>(in[inidx + 6] | (in[inidx + 7] << 8)));
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{

    unsigned int accumulate = 8;
    unsigned int codiflen = 7168;
    unsigned int fftsize = 32;
    unsigned int nofpgas = 48;
    unsigned int nopols = 2;

    unsigned int repeat = 512;

    unsigned int insize = accumulate * codiflen * nofpgas;
    unsigned int unpackedsize = accumulate * 7 * 128 * nofpgas * nopols; 

    unsigned int batchsize = unpackedsize / fftsize;
    unsigned char *codifbuffer = new unsigned char[insize];

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 codifeng{seed};
    std::uniform_int_distribution<> codifdist(0, 256);

    cout << "Generating the data..." << endl;
    cout.flush();

    for (int isamp = 0; isamp < insize; isamp++)
        codifbuffer[isamp] = codifdist(codifeng);

    cout << "The data has been generated..." << endl;
    cout.flush();

    fftwf_complex *unpacked, *ffted;
    fftwf_plan fftplan;

    unpacked = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);
    ffted = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unpackedsize);

    int sizes[1] = {fftsize};

    fftplan = fftwf_plan_many_dft(1, sizes, batchsize, unpacked, NULL, 1, fftsize, ffted, NULL, 1, fftsize, FFTW_FORWARD, FFTW_MEASURE);

    auto runstart = std::chrono::high_resolution_clock::now();

    for (int irep = 0; irep < repeat; irep++) {
        UnpackData(unpacked, codifbuffer, accumulate);
        fftwf_execute(fftplan);
    }

    auto runend = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> rundur = runend - runstart;

    cout << "Took " << rundur.count() << "s to run the unpacking and FFT" << endl;
    cout << "This is equal to " << rundur.count() * 1e+06 / (float)repeat << "us or " << rundur.count() * 1e+06 / (accumulate * 108.0) / (float)repeat << " times real time" << endl;

    fftwf_destroy_plan(fftplan);

    fftwf_free(ffted);
    fftwf_free(unpacked);

    delete [] codifbuffer;

    return 0;
}

