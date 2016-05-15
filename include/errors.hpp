#ifndef _H_PAFRB_ERRORS
#define _H_PAFRB_ERRORS

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}
inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

#endif
