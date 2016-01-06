#ifndef _H_PAFRB_CONFIG
#define _H_PAFRB_CONFIG

#include <fstream>
#include <string>

using std::string;

struct config_s {
    bool test;
    bool verbose;

    double band;                // sampling rate for each band in MHz
    double dend;
    double dstart;
    double foff;                // channel width in MHz
    double ftop;                // frequency of the top channel in MHz
    double tsamp;               // sampling time

    unsigned int beamno;        // number of beams per card
    unsigned int chunks;        // time chunks to process - testing only
    unsigned int filchans;      // number fo filterbank channels
    unsigned int freq;          // number of frequency channels to average
    unsigned int gulp;
    unsigned int nchans;        // number of 1MHz channels
    unsigned int streamno;      // number of CUDA streams for filterbanking
    unsigned int times;         // number of time samples to average

};

void default_config(config_s &config) {
    config.test = false;
    config.verbose = false;

    config.band = 1.185;
    config.dstart = 0.0;
    config.dend = 4000.0;
    config.foff = 0.0;
    config.ftop = 0.0;

    config.beamno = 1;
    config.chunks = 32;
    config.freq = 8;
    config.gulp = 131072;       // 2^17, equivalent to ~14s for 108us sampling time
    config.nchans = 192;
    config.streamno = 4;
    config.times = 4;

    config.filchans = config.nchans * 27 / config.freq;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.times;

}

void read_config(string file, config_s config) {

}

#endif
