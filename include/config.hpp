#ifndef _H_PAFRB_CONFIG
#define _H_PAFRB_CONFIG

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <heimdall/params.hpp>

struct InConfig {

    bool test;
    bool verbose;

    double band;                // sampling rate for each band in MHz
    double dmend;
    double dmstart;
    double foff;                // channel width in MHz
    double ftop;                // frequency of the top channel in MHz
    double tsamp;               // sampling time

    // NOTE: Information from the telescope
    double dec;
    double ra;

    std::string outdir;             //!< Product output directory

    std::vector<int> gpuids;        //!< GPU IDs to use
    std::vector<std::string> ips;   //!< IPs to receive the data on
    std::vector<int> killmask;      //!< Dedispersion killmask
    std::vector<int> ports;         //!< Ports to receive the data on

    unsigned int accumulate;        //!< Number of 108us complete chunks to process on the GPU at once
    unsigned int batch;             //!< FFT batch size
    unsigned int codiflen;          //!< Length (in bytes) of the single CODIF data packet (just data, no header)
    unsigned int fftsize;           //!< Single FFT size
    unsigned int filchans;          //!< Number of filterbank channels
    unsigned int freqavg;           //!< Number of frequency channels to average
    unsigned int gulp;              //!< Dedispersion gulp size
    unsigned int headlen;           //!< Length (in bytes) of the CODIF header
    unsigned int nobeams;           //!< Number of beams per card
    unsigned int nochans;           //!< Number of 1MHz channels
    unsigned int nogpus;            //!< Number of GPUs to process the data on; should be the same as gpuids.size()
    unsigned int nopols;            //!< Number of incoming polarisations
    unsigned int noports;           //!< Number of ports to receive the data on; should be the same as ports.size()
    unsigned int nostokes;          //!< Number of Stokes parameters to output
    unsigned int nostreams;         //!< Number of GPU streams used for filterbank
    unsigned int outbits;           //!< Number of filterbank bits per sample after scaling
    unsigned int record;            //!< Number of seconds to record
    unsigned int timeavg;           //!< Number of time samples to average

};

inline void SetDefaultConfig(InConfig &config) {
    config.test = false;
    config.verbose = false;

    config.accumulate = 128;
    config.band = 1.185;
    config.codiflen = 7168;
    config.dmstart = 0.0;
    config.dmend = 4000.0;
    config.ftop = 1400.0;
    config.headlen = 64;

    config.dec = 0.0;
    config.ra = 0.0;

    config.nobeams = 1;
    config.fftsize = 32;
    config.freqavg = 16;
    config.foff = (double)1.0/(double)27.0 * (double)config.freqavg;
    config.gulp = 131072;       // 2^17, equivalent to ~14s for 108us sampling time
    config.nobeams = 1;
    config.nochans = 336;
    config.nogpus = 1;
    config.nopols = 2;
    config.noports = 6;
    config.outdir = "./";
    config.record = 600;        // record ~10 minutes of data
    config.nostokes = 4;
    config.nostreams = 4;
    config.outbits = 8;
    config.timeavg = 4;

    config.batch = config.nopols * config.nochans * config.accumulate * 128 / config.fftsize;
    config.filchans = config.nochans * 27 / config.freqavg;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.timeavg;
    for (int ichan = 0; ichan < config.filchans; ichan++)
         (config.killmask).push_back((int)1);

    for (int iport = 0; iport < config.noports; iport++)
        config.ports.push_back(17100 + iport);
}

inline void ReadConfig(std::string filename, InConfig &config) {

    std::fstream inconfig(filename.c_str(), std::ios_base::in);
    std::string line;
    std::string paraname;
    std::string paravalue;

    if(inconfig) {
        while(std::getline(inconfig, line)) {
            std::istringstream ossline(line);
            ossline >> paraname >> paravalue;
            std::stringstream svalue;

            if (paraname == "ACCUMULATE") {
                config.accumulate = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "DMEND") {
                config.dmend = std::stod(paravalue);
            } else if (paraname == "DMSTART") {
                config.dmstart = std::stod(paravalue);
            } else if (paraname == "FFTSIZE") {
                config.fftsize = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "FREQAVG") {
                config.freqavg = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "DEDISPGULP") {
                config.gulp = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "GPUIDS") {
                std::stringstream svalue(paravalue);
                std::string sep;
                while(std::getline(svalue, sep, ','))
                    config.gpuids.push_back(std::stoi(sep));
            } else if (paraname == "IPS") {
                std::stringstream svalue(paravalue);
                std::string sep;
                while(std::getline(svalue, sep, ','))
                    config.ips.push_back(sep);
            } else if (paraname == "NO1MHZCHANS") {
                config.nochans = (unsigned int)(std::stoi(paravalue));
                config.batch = config.nochans;
            } else if (paraname == "NOBEAMS") {
                config.nobeams = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NOGPUS") {
                config.nogpus = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NOPOLS") {
                config.nopols = std::stoi(paravalue);
            } else if (paraname == "NOSTOKES") {
                config.nostokes = std::stoi(paravalue);
            } else if (paraname == "NOSTREAMS") {
                config.nostreams = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "OUTBITS") {
                config.outbits = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "PORTS") {
                std::stringstream svalue(paravalue);
                std::string sep;
                // NOTE: Need to remove the ports added as default parameters
                config.ports.clear();
                while(std::getline(svalue, sep, ','))
                    config.ports.push_back(std::stoi(sep));
                config.noports = config.ports.size();
            } else if (paraname == "TIMEAVG") {
                config.timeavg = (unsigned int)(std::stoi(paravalue));
            } else {
                std::cerr << "Error: unrecognised parameter: " << paraname << std::endl;
            }
        }
    } else {
        std::cerr << "Error opening the configuration file!!\n Will use the default configuration instead." << std::endl;
    }

    // NOTE: Need to restart these values, as they depend on variables that can be enter through configuration file
    config.batch = config.nopols * config.nochans * config.accumulate * 128 / config.fftsize;
    config.filchans = config.nochans * 27 / config.freqavg;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.timeavg;

    inconfig.close();
}

inline void SetSearchParams(hd_params &params, InConfig config)
{
    params.verbosity       = 0;
    #ifdef HAVE_PSRDADA
    params.dada_id         = 0;
    #endif
    params.sigproc_file    = NULL;
    params.yield_cpu       = false;
    params.nsamps_gulp     = config.gulp;
    // TODO: This is no longer being used
    params.dm_gulp_size    = 2048;//256;    // TODO: Check that this is good
    params.baseline_length = 2.0;
    params.beam            = 0;
    params.override_beam   = false;
    params.nchans          = config.filchans;
    params.dt              = config.tsamp;
    params.f0              = config.ftop;
    params.df              = -abs(config.foff);    // just to make sure it is negative
    // no need for dm params as the code will not do it
    params.dm_min          = config.dmstart;
    params.dm_max          = config.dmend;
    params.dm_tol          = 1.25;
    params.dm_pulse_width  = 40;//e-6; // TODO: Check why this was here
    params.dm_nbits        = 32;//8;
    params.use_scrunching  = false;
    params.scrunch_tol     = 1.15;
    params.rfi_tol         = 5.0;//1e-6;//1e-9; TODO: Should this be a probability instead?
    params.rfi_min_beams   = 8;
    params.boxcar_max      = 4096;
    params.detect_thresh   = 6.0;
    params.cand_sep_time   = 3;
    // Note: These have very little effect on the candidates, but could be important
    //         to capture (*rare*) coincident events.
    params.cand_sep_filter = 3;  // Note: filter numbers, not actual width
    params.cand_sep_dm     = 200; // Note: trials, not actual DM
    params.cand_rfi_dm_cut = 1.5;
    //params.cand_min_members = 3;

    // TODO: This still needs tuning!
    params.max_giant_rate  = 0;      // Max allowed giants per minute, 0 == no limit

    params.min_tscrunch_width = 4096; // Filter width at which to begin tscrunching

    params.num_channel_zaps = 0;
    params.channel_zaps = NULL;

    params.coincidencer_host = NULL;
    params.coincidencer_port = -1;

    // TESTING
    //params.first_beam = 0;
    params.beam_count = 1;
    params.gpu_id = 0;
    params.utc_start = 0;
    params.spectra_per_second = 0;
    params.output_dir = ".";
}
#endif
