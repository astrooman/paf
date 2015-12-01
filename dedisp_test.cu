#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <dedisp.h>
#include <DedispPlan.hpp>

using std::cout;
using std::endl;
using std::string;
using std::vector;



int main(int argc, char *argv[]) {

    // these are just very rough default values
    float tsamp{108e-06};      // sampling time in seconds
    float ftop{1700};          // highest frequency channel in MHz
    float foff{0.5291};        // channel bandwidth in MHz
    unsigned int nchans{567};
    unsigned int gulp{262144};

    float dstart{0.0};
    float dend{2000.0};

    if (argc >= 2) {
        for (int ii = 0; ii < argc; ii++) {
            if (string(argv[ii]) == "-t") {
                ii++;
                tsamp = atof(argv[ii]);
            } else if (string(argv[ii]) == "-f") {
                ii++;
                ftop = atof(argv[ii]);
            } else if (string(argv[ii]) == "-o") {
                ii++;
                foff = atof(argv[ii]);
            } else if (string(argv[ii]) == "-n") {
                ii++;
                nchans = atoi(argv[ii]);
            } else if (string(argv[ii]) == "-g") {
                ii++;
                gulp = atoi(argv[ii]);
            } else if (string(argv[ii]) == "-h") {
                cout << "Options:\n"
                        << "\t -t - sampling time in seconds\n"
                        << "\t -f - the frequency of the top channel in MHz\n"
                        << "\t -o - channel bandwidth in MHz\n"
                        << "\t -n - the number of frequency channels\n"
                        << "\t -g - gulp size\n"
                        << "\t -h - print out this message\n\n";
		exit(EXIT_SUCCESS);
            }
        }
    }

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 reng{seed};
    std::normal_distribution<float> rdist(128.0, 32.0);

    unsigned char *data_in = new unsigned char[nchans * gulp * 2];
    for (int ii = 0; ii < nchans * gulp * 2; ii++)
        data_in[ii] = (unsigned char)rdist(reng);

    unsigned char *data_out= new unsigned char[gulp];

    std::chrono::time_point<std::chrono::system_clock> initstart, initend;
    std::chrono::duration<double> initelapsed;
    std::chrono::time_point<std::chrono::system_clock> dedistart, dediend;
    std::chrono::duration<double> dedielapsed;

    cout << "Initialising...\n";
    initstart = std::chrono::system_clock::now();
    DedispPlan dedisp(nchans, tsamp, ftop, foff);
    dedisp.generate_dm_list(dstart, dend, (float)64.0, (float)1.10);
    initend = std::chrono::system_clock::now();
    initelapsed = initend - initstart;

    cout << "Dedispersing...\n";
    dedistart = std::chrono::system_clock::now();
    dedisp.execute(gulp, data_in, sizeof(float) * 8, data_out, sizeof(unsigned char) * 8, 0);
    dediend = std::chrono::system_clock::now();
    dedielapsed = dediend - dedistart;

    cout << "Gulp size " << tsamp * gulp << "s\n";
    cout << "Initialisation time " << initelapsed.count() <<"s\n";
    cout << "Dedispersion time " << dedielapsed.count() << "s\n";
    if (dedielapsed.count() >= tsamp * gulp)
        cout << "We have a problem!!\n";
}
