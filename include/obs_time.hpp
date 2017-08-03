#ifndef _H_PAFRB_OBS_TIME
#define _H_PAFRB_OBS_TIME

struct ObsTime {
    int refepoch;            // reference epoch at the start of the observation
    int refsecond;           // seconds from the reference epoch at the start of the observation
    int refframe;                // frame number from the start of the observation
};

#endif
