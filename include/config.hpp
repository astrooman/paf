#ifndef _H_PAFRB_CONFIG
#define _H_PAFRB_CONFIG

#include <fstream>
#include <string>

#include <heimdall/params.hpp>

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

    unsigned int batch;
    unsigned int beamno;        // number of beams per card
    unsigned int chunks;        // time chunks to process - testing only
    unsigned int fftsize;
    unsigned int filchans;      // number fo filterbank channels
    unsigned int freqavg;          // number of frequency channels to average
    unsigned int gulp;
    unsigned int nchans;        // number of 1MHz channels
    unsigned int ngpus;         // number of GPUs to use
    unsigned int npol;
    unsigned int port;
    unsigned int stokes;        // number of Stokes parameters to output
    unsigned int streamno;      // number of CUDA streams for filterbanking
    unsigned int timesavg;         // number of time samples to average

};

inline void default_config(config_s &config) {
    config.test = false;
    config.verbose = false;

    config.band = 1.185;
    config.dstart = 0.0;
    config.dend = 4000.0;
    config.foff = 0.50;
    config.ftop = 1400.0;

    config.beamno = 1;
    config.chunks = 32;
    config.fftsize = 32;
    config.freqavg = 8;
    config.gulp = 131072;       // 2^17, equivalent to ~14s for 108us sampling time
    config.nchans = 7;
    config.ngpus = 3;
    config.npol = 2;
    config.stokes = 4;
    config.streamno = 4;
    config.timesavg = 4;

    config.batch = config.nchans;
    config.filchans = config.nchans * 27 / config.freqavg;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.timesavg;

}

inline void read_config(string file, config_s config) {

}

inline void set_search_params(hd_params &params, config_s config)
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
    params.dm_min          = config.dstart;
    params.dm_max          = config.dend;
    params.dm_tol          = 1.25;
    params.dm_pulse_width  = 40;//e-6; // TODO: Check why this was here
    params.dm_nbits        = 32;//8;
    params.use_scrunching  = false;
    params.scrunch_tol     = 1.15;
    params.rfi_tol         = 5.0;//1e-6;//1e-9; TODO: Should this be a probability instead?
    params.rfi_min_beams   = 8;
    params.boxcar_max      = 4096;//2048;//512;
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