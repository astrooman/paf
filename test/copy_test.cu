#include <chrono>
#include <iostream>

#include <cufft.h>

using std::cout;
using std::endl;

#define XSIZE = 7
#define YSIZE = 128
#define ZSIZE = 48

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

int main(int argc, char *argv[])
{

    float alloc_elapsed;
    cudaEvent_t alloc_start;
    cudaEvent_t alloc_end;

    cudaEventCreate(&alloc_start);
    cudaEventCreate(&alloc_end);

    char *h_in = new char[8 * XSIZE * YSIZE * ZSIZE];
    for (int ii = 0; ii < 8 * XSIZE * YSIZE * ZSIZE)
        h_in[ii] = ii % 7;

    cudaChannelFormatDesc cdesc;
    cudaExtent volume;
    cudaMemcpy3DParms params = {0};

    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());
    volume = make_cudaExtent(XSIZE, YSIZE, ZSIZE);
    cudaCheckError(cudaPeekAtLastError());
    cudaArray *d_array;

    cudaEventRecord(alloc_start, 0);
    cudaCheckError(cudaMalloc3DArray(&d_array, &cdesc, volume));
    cudaCheckError(cudaBindTextureToArray(tex, d_array));
    cudaEventRecord(alloc_end, 0);
    cudaEventSynchronize(alloc_end);
    cudaEventElapsedTime(&alloc_elapsed, alloc_start, alloc_end);

    cout << "3D alloc: " << alloc_elapsed << "ms" << endl;

    params.extent = volume;
    params.dstArray = d_array;
    params.kind = cudaMemcpyHostToDevice;
    params.srcPtr = make_cudaPitchedPtr((void*)h_in, XSIZE * 8, XSIZE * 8, YSIZE);

    tex.filterMode = cudaFilterModePoint;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.addressMode[2] = cudaAddressModeClamp;

    float copy_elapsed;
    cudaEvent_t copy_start;
    cudaEvent_t copy_end;

    cudaEventCreate(&copy_start);
    cudaEventCreate(&copy_end);

    cudaEventRecord(copy_start,0);
    cudaCheckError(cudaMemcpy3D(&params));
    cudaEventrecord(copy_end.0);
    cudaEventSynchronize(copy_end);
    cudaEventElapsedTime(&copy_elapsed, copy_start, copy_end);

    cout << "3D memcpy: " << copy_elapsed << "ms" << endl;

    cudaDeviceSynchronize();
    cudaUnbindTexture(tex);

    cudaFreeArray(d_array);
    cudaFree(d_out);
    delete [] h_in;

    return 0;
}
