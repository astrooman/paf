#include <chrono>
#include <cstdlib>
#include <iomanip>
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
    unsigned int gulp{1024};

    float dstart{0.0};
    float dend{0.0};

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
	    } else if (string(argv[ii]) == "--ds") {
		ii++;
		dstart = atof(argv[ii]);
	    } else if (string(argv[ii]) == "--de") {
		ii++;
		dend = atof(argv[ii]);
            } else if (string(argv[ii]) == "-h") {
                cout << "Options:\n"
                        << "\t -t - sampling time in seconds\n"
                        << "\t -f - the frequency of the top channel in MHz\n"
                        << "\t -o - channel bandwidth in MHz\n"
                        << "\t -n - the number of frequency channels\n"
                        << "\t -g - gulp size\n"
			<< "\t --ds - start dm\n"
			<< "\t --de - end dm\n"
                        << "\t -h - print out this message\n\n";
		exit(EXIT_SUCCESS);
            }
        }
    }

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 reng{seed};
    std::normal_distribution<float> rdist(128.0, 32.0);

    std::chrono::time_point<std::chrono::system_clock> initstart, initend;
    std::chrono::duration<double> initelapsed;
    std::chrono::time_point<std::chrono::system_clock> dedistart, dediend;
    std::chrono::duration<double> dedielapsed;

    cout << "Initialising...\n";
    initstart = std::chrono::system_clock::now();
    // ftop and fofff in MHz
    DedispPlan dedisp(nchans, tsamp, ftop, foff);
    dedisp.generate_dm_list(dstart, dend, (float)64.0, (float)1.10);
    initend = std::chrono::system_clock::now();
    initelapsed = initend - initstart;

    cout << "Filling the array...\n";
    cout << "Will generate " << nchans << " channels with " << gulp + dedisp.get_max_delay() << " time samples each\n";
    unsigned char *data_in = new unsigned char[nchans * gulp + nchans * (int)dedisp.get_max_delay()];
    cout << std::setprecision(2) << std::fixed;
    for (int ii = 0; ii < nchans * gulp + nchans * (int)dedisp.get_max_delay(); ii++) {
        data_in[ii] = (unsigned char)rdist(reng);
	cout << (double)ii / (double)(nchans * gulp + nchans * (int)dedisp.get_max_delay()) * (double)100 << "% complete\r";
	cout.flush();
    }
    std::cin.get();
    unsigned char *data_out= new unsigned char[gulp * dedisp.get_dm_count()];

    cout << "Dedispersing...\n";
    dedistart = std::chrono::system_clock::now();
    dedisp.execute(gulp + dedisp.get_max_delay(), data_in, sizeof(unsigned char) * 8, data_out, sizeof(unsigned char) * 8, 0);
    dediend = std::chrono::system_clock::now();
    dedielapsed = dediend - dedistart;

    cout << "Gulp size " << tsamp * gulp << "s\n";
    cout << "Initialisation time " << initelapsed.count() <<"s\n";
    cout << "Dedispersion time " << dedielapsed.count() << "s\n";
    if (dedielapsed.count() >= tsamp * gulp)
        cout << "We have a problem!!\n";
    std::cin.get();
    delete [] data_in;
    delete [] data_out;
}
