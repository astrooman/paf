#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda.h>

#include "errors.hpp"

using std::cout;
using std::endl;

__global__ void scale(float* in, float* out, unsigned int nchans, unsigned int time_samples)
{
    // call one block with 32 threads
    // be careful when processing total sizes that cannot be divided by 32
    // or make sure the total size can be divided by 32 when allocating
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float nrec = 1.0f / (float)nchans;
    float mean;
    float std;

    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int start = 0;
    float nrec1 = 1.0f / (float)(nchans -1.0f);
    for (int chunk = 0; chunk < (time_samples / threads); chunk++) {
        mean = 0.0f;
        std = 0.0f;
        start = chunk * threads * nchans;

        for (int ii = 0; ii < nchans; ii++) {
            mean += in[start + idx * nchans + ii] * nrec;
            //printf("%d\n", mean);
        }

        for (int jj = 0; jj < nchans; jj++) {
            std += (in[start + idx * nchans + jj] - mean) * (in[start + idx * nchans + jj] - mean);
        }
        std *= nrec1;

        //printf("%i: %i, %f, %f, %f\n", idx, nchans, nrec, mean, std);

        float stdrec = rsqrtf(std);

        for (int kk = 0; kk < nchans; kk++) {
            out[start + idx * nchans + kk] = ((in[start + idx * nchans + kk] - mean) * stdrec) * 32.0f + 64.0f;
            if (out[start + idx * nchans + kk] < 0.0f)
                out[start + idx * nchans + kk] = 0.0f;
        }
    }

}

int main(int argc, char *argv[])
{
        std::stringstream oss;
        int time_samp{0};
        time_samp = atoi(argv[2]);
        std::string inname, outname;
        cout << "Reading file: " << inname << endl;
        std::ifstream input_file(argv[1], std::ios_base::in | std::ios_base::binary);

        std::string read_param;
        char field[60];

        int strlen;

        input_file.seekg(369, input_file.beg);

        char *head = new char[4];

        input_file.read(head, 4);

        for (int ii = 0; ii < 4; ii++)
                cout << head[ii] << " ";

        cout << endl << endl;
        cout.flush();
        size_t to_read = 567 * time_samp * 4;

        float *data = new float[567 * time_samp];
        float *scaled = new float[567 * time_samp];
        cout << "Reading some data now..." << endl;
        input_file.read(reinterpret_cast<char*>(data), to_read);
        input_file.close();

        for (int ii = 0; ii < 21; ii++)
                cout << data[ii] << " ";
        cout << endl << endl;

        float *d_data;
        float *d_scaled;
        cudaCheckError(cudaMalloc((void**)&d_data, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_scaled, 567 * time_samp * sizeof(float)));

        cudaCheckError(cudaMemcpy(d_data, data, 567 * time_samp * sizeof(float), cudaMemcpyHostToDevice));
        cout << "Starting the kernel...\n";
        cout.flush();
        for (int ii = 0; ii < 1; ii++)
            scale<<<2, 32, 0>>>(d_data, d_data, 567, time_samp);       
        cudaDeviceSynchronize(); 
        cudaCheckError(cudaMemcpy(scaled, d_data, 567 * time_samp * sizeof(float), cudaMemcpyDeviceToHost));

        for (int ii = 0; ii < 21; ii++)
            cout << scaled[ii] << endl;
        cout << endl << endl;

        cout << "Saving the output file...\n";

        std::ofstream output_file(argv[3], std::ios_base::out | std::ios_base::trunc);

        if (output_file) {
            for (int ii = 0; ii < 567 * time_samp; ii++)
                output_file << scaled[ii] << endl;
        }
        output_file.close();

        delete [] data;
        delete [] scaled;
        cudaCheckError(cudaFree(d_data));
        cudaCheckError(cudaFree(d_scaled));

        return 0;
}

