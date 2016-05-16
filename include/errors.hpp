#ifndef _H_PAFRB_ERRORS
#define _H_PAFRB_ERRORS

#define cudaCheckError(myerror) {checkGPU((myerror), __FILE__, __LINE__);}
#define cufftCheckError(mymsg) {checkFFT((mymsg), __FILE__, __LINE__);}

inline void checkGPU(cudaError_t code, const char *file, int line) {

    if (code != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(code) << " in file " << file << ", line " << line << endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}
inline const char *cufftGetErrorString(cufftResult_t msg) {

    switch(msg) {
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPATIBLE_PARAMETER_LIST:
            return "CUFFT_INCOMPATIBLE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";
    }
}

inline void checkFFT(cufftResult_t msg, const char *file, int line) {

    if (msg != CUFFT_SUCCESS) {
        cout << "CUFFT error: " <<  << " in file " << file << ", line " << line << endl;
        exit(EXIT_FAILURE);
        // TODO: throw exception instead of exiting
    }

}

#endif
