#include <thrust/host_vector.hpp>
#include <thrust/device_vector.hpp>

int main(int argc, char *argv[])
{

    thrust::device_vector<float> tv1(567 * 4);

    // 567 channels * 4 Stokes parameters
    float *hv1 = new float[567 * 4];

    for (int ii = 0; ii < 567 * 4; ii++)
        hv1[ii] = (float)ii;

    float* dv1;

    cudaMalloc((void**)&dv1, 567 * 4 * sizeof(float));
    cudaMemcpy(dv1, hv1, 567 * 4 * sizeof(float), cudaMemcpyHostToDevice);

    thrust::copy(hv1, hv1 + 567 * 4, tv1.begin());

    return 0;
}
