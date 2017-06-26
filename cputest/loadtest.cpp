#include <iostream>
#include <fftw3.h>

#include "immintrin.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{

    float myarray[8];

    for (int ifloat = 0; ifloat < 8; ++ifloat) {
        myarray[ifloat] = (float)ifloat / 2.0f + 1.0f;
    }

    cout << "Original array: " << endl;

    for (int ifloat = 0; ifloat < 8; ++ifloat) 
        cout << myarray[ifloat] << endl;

    cout << endl << endl;

    __m256 loadf = _mm256_loadu_ps(myarray);

    cout << "Loaded array: " << endl;

    for (int ifloat = 0; ifloat < 8; ++ifloat)
        cout << *((float*)&loadf + ifloat) << endl;
    __m256i mask = _mm256_set_epi32(0xf0000000,0,0,0,0,0,0,0xffffffff);
    //__m256i mask = _mm256_set_epi16(0,0,0,0,0,0,0,0,0,0,0,0,0xf000,0,0xf000,0);
    __m256 loadfm = _mm256_maskload_ps(myarray, mask);

    cout << "Masked loaded array: " << endl;

    for (int ifloat = 0; ifloat < 8; ++ifloat)
        cout << *((float*)&loadfm + ifloat) << endl;
    
    cout << "And now for the complex numbers fun!" << endl;

    fftwf_complex *mycomplex;
    mycomplex = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 8);

    for (int icomp = 0; icomp < 8; ++icomp) {
        mycomplex[icomp][0] = (float)icomp + 0.5;
        mycomplex[icomp][1] = (float)icomp + 1.5;
    }

    cout << "Original complex array: " << endl;
 
    for (int icomp = 0; icomp < 8; ++icomp)
        cout << mycomplex[icomp][0] << " + i * " << mycomplex[icomp][1] << endl;

    cout << endl << endl;

    __m256 loadc1 = _mm256_loadu_ps(reinterpret_cast<float*>(mycomplex));
    __m256 loadc2 = _mm256_loadu_ps(reinterpret_cast<float*>(mycomplex + 4));

    cout << "Loaded complex array: " << endl;

    for (int icomp = 0; icomp < 8; ++icomp)
        cout << *((float*)&loadc1 + icomp) << endl;

    for (int icomp = 0; icomp < 8; ++icomp)
        cout << *((float*)&loadc2 + icomp) << endl;

    __m256 pow2c1 = _mm256_mul_ps(loadc1, loadc1);
    __m256 pow2c2 = _mm256_mul_ps(loadc2, loadc2);
    cout << "Squared loaded complex array: " << endl;

    for (int ipow = 0; ipow < 8; ++ipow)
        cout << *((float*)&pow2c1 + ipow) << endl;
   
    for (int ipow = 0; ipow < 8; ++ipow)
        cout << *((float*)&pow2c2 + ipow) << endl;

    cout << endl << endl;

    __m256 pow2hadd = _mm256_hadd_ps(pow2c1, pow2c2);

    cout << "Horizontally added power array: " << endl;

    for (int iadd = 0; iadd < 8; ++iadd)
        cout << *((float*)&pow2hadd + iadd) << endl;
 
    return 0;

}
