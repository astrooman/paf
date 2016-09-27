#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda.h>

#include "errors.hpp"

using std::cout;
using std::endl;


__global__ void scale_factors(float* in, float* means, float* stdevs, unsigned int nchans, unsigned int times)
{
    // call one block with n * 32 threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float nrec = 1.0f / (float)times;
    float mean;
    float stdev;

    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int start = 0;
    float nrec1 = 1.0f / (float)(times - 1.0f);

    // channels per thread
    int cpt = (int)(nchans / threads);
    int rem = nchans - cpt * threads;
    int jump = nchans;

    for (int chan = 0; chan < cpt; chan++) {
        mean = 0.0f;
        stdev = 0.0;
        start = idx * cpt + chan;

        for (int samp = 0; samp < times; samp++) {
            mean += in[start + samp * jump];
        }
        mean *= nrec;
        means[start] = mean;

        for (int samp = 0; samp < times; samp++) {
            stdev += (in[start + samp * jump] - mean) * (in[start + samp * jump] - mean);
        }
        stdev *= nrec1;
        stdevs[start] = sqrtf(stdev);

    }

    // the last thread does the remainder calculations
    // this part can slow thigns down significantly
    // TODO: come up with a slightly better solution for handling the remainder
    if (idx == (threads - 1)) {
        printf("Getting the remainder...\n");
        for (int chan = 0; chan < rem; chan++) {
            mean = 0.0f;
            stdev = 0.0;
            start = cpt * threads + chan;

            for (int samp = 0; samp < times; samp++) {
                mean += in[start + samp * jump] * nrec;
            }
            means[start] = mean;

            for (int samp = 0; samp < times; samp++) {
                stdev += (in[start + samp * jump] - mean) * (in[start + samp * jump] - mean);
            }
            stdev *= nrec1;
            stdevs[start] = sqrtf(stdev);
        }
    }
}

__global__ void scale(float *in, float *out, float *means, float *stdevs, unsigned int nchans, unsigned int times)
{
    // call one block with n * 32 threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float nrec = 1.0f / (float)times;
    float mean;
    float stdev;

    unsigned int threads = blockDim.x * gridDim.x;
    unsigned int start = 0;
    float nrec1 = 1.0f / (float)(times - 1.0f);

    // channels per thread
    int cpt = (int)(nchans / threads);
    int rem = nchans - cpt * threads;
    int jump = nchans;

    for (int chan = 0; chan < cpt; chan++) {
        
        start = idx * cpt + chan;

        for (int samp = 0; samp < times; samp++) {
            out[start + nchans * samp] = ((in[start + nchans * samp] - means[start]) / stdevs[start]) * 32.0f + 64.0f;
        }
    }
   
    if (idx == (threads -1)) {
        for (int chan = 0; chan < rem; chan++) {
            start = cpt * threads + chan;
            
            for (int samp = 0; samp < times; samp++) {
                out[start + nchans * samp] = ((in[start + nchans * samp] - means[start]) / stdevs[start]) * 32.0f + 64.0f;
            }
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
        float *means = new float[567];
        float *stdevs = new float[567];
        cout << "Reading some data now..." << endl;
        input_file.read(reinterpret_cast<char*>(data), to_read);
        input_file.close();

        for (int ii = 0; ii < 21; ii++)
                cout << data[ii] << " ";
        cout << endl << endl;

        float *d_data;
        float *d_scaled;
        float *d_means;
        float *d_stdevs;
        cudaCheckError(cudaMalloc((void**)&d_data, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_scaled, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_means, 567 * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_stdevs, 567 * sizeof(float)));

        cudaCheckError(cudaMemcpy(d_data, data, 567 * time_samp * sizeof(float), cudaMemcpyHostToDevice));
        cout << "Starting the kernel...\n";
        cout.flush();
        for (int ii = 0; ii < 5; ii++)
            scale_factors<<<1,32,0,0>>>(d_data, d_means, d_stdevs, 567, time_samp);
        cudaDeviceSynchronize(); 

        for (int ii = 0; ii < 5; ii++)
            scale<<<1,32,0,0>>>(d_data, d_scaled, d_means, d_stdevs, 567, time_samp);
        cudaDeviceSynchronize();

        cudaCheckError(cudaMemcpy(means, d_means, 567 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(stdevs, d_stdevs, 567 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(scaled, d_scaled, 567 * time_samp * sizeof(float), cudaMemcpyDeviceToHost));

/*        cout << "MEANS\n";
        for (int ii = 0; ii < 567; ii++)
            cout << means[ii] << endl;

        cout << "STDEVS:\n";
        for (int ii = 0; ii < 567; ii++)
            cout << stdevs[ii] << endl;
*/
        cout << "Saving the output file " << argv[3] << "...\n";

        std::ofstream output_file(argv[3], std::ios_base::out | std::ios_base::trunc);

        if (output_file) {
            for (int ii = 0; ii < 567 * time_samp; ii++)
                output_file << scaled[ii] << endl;
        }
        output_file.close();

        delete [] data;
        delete [] scaled;
        delete [] means;
        delete [] stdevs;
        cudaCheckError(cudaFree(d_data));
        cudaCheckError(cudaFree(d_scaled));
        cudaCheckError(cudaFree(d_means));
        cudaCheckError(cudaFree(d_stdevs));

        return 0;
}

