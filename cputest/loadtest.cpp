#include <iostream>

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
    

    return 0;

}
