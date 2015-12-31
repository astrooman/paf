#include <fstream>
#include <iostream>

#define HEADER 64   // header is 64 bytes long

struct header_s {
    // thigs are listed in the order they appear in the vdif header
    // WORD 0
    int ref_s;      // seconds from reference epoch
    bool comp;      // complex data flag: real = 0, complex = 1
    bool invalid;   // invalid data: valid = 0, invalid = 1
    // WORD 1
    int frame_no;   // data frame within the current period
    // WORD 2
    int arr_len;    // data array length in units of 8 bytes
    int in_bits;    // bits per sample
    int version;    // vdif version number; 1 for VDIF2 (doesn't make much sense)
    // WORD 3
    int station;    // station ID
    int rep;        // sample representation: 0 - offset binary, 1 - 2's compliment, 2 - IEEE floating point
    int epoch;      // reference epoch
    // WORD 4
    int nchans;      // number of channels - 1 (why the hell minus 1?!)
    int block_len;  // sample block length - 1 in units of 8 bytes (must be large enough to hold
                    // at least one complete sample: sample size * nchans (* 2 for complex data))
    // WORD 5
    int group;      // group ID
    int thread;     // thread ID
    // WORD 6
    int period;     // in seconds - number  seconds during which there is exactly an integral number of sample periods
    // WORD 7 - reserved for future use
    // WORD 8
    int sipp_m;     // sample intervals per period (MSB)
    // WORD 9
    int sipp_l;     // sample intervals per period (LSB)
    // WORD 10
    int synch;      // 32-bit synchronisation word (‘0xadeadbee’), always present
    // WORD 11 - reserved for future use
    // WORD 12
    int edv;        // extended data version
    // WORD 12, 13, 14, 15 - extended user data
};

// should read header from the data packet
void get_header(unsigned char* packet)
{
    header_s head;

    ref_s = (int)(packet[0] | (packet[1] << 8) | (packet[2] << 16) | ((packet[3] << 24) & 0x3f000000));
    comp = (bool)((packet[3] & 0x40) >> 6);
    invalid = (bool)((packet[3] & 0x80) >> 7);
    frame_no = (int)(packet[4] | (packet[5] << 8) | (packet[6] << 16) | (packet[7] << 24));
    arr_len = (int)(packet[8] | (packet[9] << 8) | (packet[10] << 16));
    in_bits = (int)(packet[11] & 0x1f);         // 0x1f -> 00011111
    version = (int)((packet[11] & 0xe0) >> 5);         // 0xe0 -> 11100000
    station = (int)(packet[12] | (packet[13] << 16));
    rep = (int)((packet[14] >> 6 ) | ((packet[15] & 0x3) << 2));      // 0x3 -> 00000011
    epoch = (int)((packet[15] & 0xfc) >> 2);     // 0xfc -> 11111100
    nchans = (int)(packet[16] | (packet[17] << 8));
    block_len = (int)(packet[18] | (packet[19] << 8));
    group = (int)(packet[20] | (packet[21] << 8));
    thread = (int)(packet[22] | (packet[23] << 8));
    period = (int)(packet[24] | (packet[25] << 8));
    synch = (int)(packet[40] | (packet[41] << 8) | (packet[42] << 16) | (packet[43] << 24));

    long sipp = (packet[32] | (packet[33] << 8) | (packet[34] << 16) | (packet[35] << 24)
                    | (packet[36] << 32) | (packet[37] << 40) | (packet[38] << 48) | (packet[39] << 56));

    cout << "VDIF version " << version + 1 << endl;
    cout << "Seconds from the reference epoch: " << ref_s << endl;
    cout << "Period: " << period << endl;
    cout << "Number of channels: " << nchans + 1 << endl;
    cout << "Data frame within the current period: " << frame_no << endl;

}
