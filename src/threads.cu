#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "config.hpp"
#include "errors.hpp"
#include "ober_pool.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::thread;
using std::vector;

int main(int argc, char *argv[])
{
    string config_file;
    InConfig config;
    SetDefaultConfig(config);

    // too many parameters to load as arguments - use config file
    if (argc >= 2) {
        for (int iarg = 0; iarg < argc; iarg  ++) {
            if (string(argv[iarg]) == "--config") {      // configuration file
                iarg++;
                config_file = string(argv[iarg]);
                ReadConfig(config_file, config);
            } else if (string(argv[iarg]) == "-c") {      // the number of chunks to process
                iarg++;
                config.chunks = atoi(argv[iarg]);
	    } else if (string(argv[iarg]) == "-r") {
                iarg++;
                config.record = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-s") {     // the number of streams to use
                iarg++;
                config.streamno = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-b") {     // the number of beams to accept the data from
                iarg++;
                config.beamno = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-t") {     // the number of time sample to average
                iarg++;
                config.timeavg = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-f") {     // the number of frequency channels to average
                iarg++;
                config.freqavg = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-n") {    // the number of GPUs to use
                iarg++;
                config.ngpus = atoi(argv[iarg]);
            } else if (string(argv[iarg]) == "-o") {    // output directory for the filterbank files
                iarg++;
                struct stat chkdir;
                if (stat(argv[iarg], &chkdir) == -1) {
                    cerr << "Stat error" << endl;
                } else {
                    bool isdir = S_ISDIR(chkdir.st_mode);
                    if (isdir)
                        config.outdir = string(argv[iarg]);
                    else
                        cout << "Output directory does not exist! Will use default directory!";
                }
            } else if (string(argv[iarg]) == "--gpuid") {
                for (int igpu = 0; igpu < config.ngpus; igpu++) {
                    iarg++;
                    config.gpuids.push_back(atoi(argv[iarg]));
                }
            } else if (string(argv[iarg]) == "--ip") {
                for (int iip = 0; iip < config.ngpus; iip++) {
                    iarg++;
                    config.ips.push_back(string(argv[iarg]));
                }
            } else if (string(argv[iarg]) == "-v") {
                config.verbose = true;
            } else if ((string(argv[iarg]) == "-h") || (string(argv[iarg]) == "--help")) {
                cout << "Options:\n"
                        << "\t -h --help - print out this message\n"
                        << "\t --config <file name> - configuration file\n"
                        << "\t -b - the number of beams to process\n"
                        << "\t -c - the number of chunks to process\n"
                        << "\t -f - the number of frequency channels to average\n"
                        << "\t -n - the number of GPUs to use\n"
                        << "\t -o <directory> - output directory\n"
                        << "\t -r - the number of seconds to record\n"
                        << "\t -s - the number of CUDA streams per GPU to use\n"
                        << "\t -t - the number of time samples to average\n"
                        << "\t -v - use verbose mode\n"
                        << "\t --gpuid - GPU IDs to use - the number must be the same as 'n'\n"
                        << "\t --ip - IPs to listen to - the number must be the same as 'n'\n\n";
                exit(EXIT_SUCCESS);
            }
        }

    }

    // TODO: Expand the configuration output a bit
    if (config.verbose) {
        cout << "Starting up. This may take few seconds..." << endl;

        cout << "This is the configuration used:" << endl;
        cout << "\t - the number of GPUs to use: " << config.ngpus << endl;
        int devcount{0};
        cudaCheckError(cudaGetDeviceCount(&devcount));
        if (config.ngpus > devcount) {
            cout << "You can't use more GPUs than you have available!" << endl;
            config.ngpus = devcount;
        }
        cout << "\t - the number of worker streams per GPU: " << config.streamno << endl;
        cout << "\t - the IP addresses to listen on:" << endl;
        for (int iip = 0; iip < config.ips.size(); iip++) {
            cout << "\t\t * " << config.ips[iip] << endl;
        }
        cout << "\t - gulp size: " << config.gulp << endl;
        cout << "\t - frequency averaging: " << config.freqavg << endl;
        cout << "\t - time averaging: " << config.timeavg << endl;
    }

    OberPool mypool(config);

    cudaDeviceReset();

    return 0;
}
