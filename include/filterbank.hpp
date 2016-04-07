#ifndef _H_PAFRB_FILTERBANK
#define _H_PAFRB_FILTERBANK

#include <cstring>
#include <fstream>
#include <string>
#include <sstream>

void save_filterbank(float **ph_filterbank, size_t nsamps, size_t start)
{
    std::ostringstream oss;
    oss.str("");
    std::string filename;
    filename = oss.str() + ".dat";
    std::fstream outfile(filename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    int length{0};
    char field[60];

    // save only when the stream has been opened correctly
    if(outfile) {

        length = 12;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "HEADER_START");
        outfile.write(field, length * sizeof(char));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "telescope_id");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.tel_id, sizeof(int));

        length = 11;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "rawdatafile");
        outfile.write(field, length * sizeof(char));
        length = head.raw_file.size();
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, head.raw_file.c_str());
        outfile.write(field, length * sizeof(char));

        length = 11;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "source_name");
        outfile.write(field, length * sizeof(char));
        length = head.source_name.size();
        strcpy(field, head.source_name.c_str());
        outfile.write((char*)&length, sizeof(int));
        outfile.write(field, length * sizeof(char));

        length = 10;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "machine_id");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.machine_id, sizeof(int));

        length = 9;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "data_type");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.data_type, sizeof(int));

        length = 8;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "az_start");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.az, sizeof(double));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "za_start");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.za, sizeof(double));

        length = 7;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "src_raj");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.ra, sizeof(double));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "src_dej");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.dec, sizeof(double));

        length = 6;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "tstart");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.tstart, sizeof(double));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "nchans");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.nchans, sizeof(int));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "nbeams");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.nbeams, sizeof(int));

        length = 5;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "tsamp");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.tsamp, sizeof(int));

        // bits per time sample
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "nbits");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.nbits, sizeof(int));

        // reference dm - not really sure what it does
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "refdm");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.rdm, sizeof(int));

        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "ibeam");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.ibeam, sizeof(int));

        length = 4;
        // the frequency of the top channel
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "fch1");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.fch1, sizeof(int));

        // channel bandwidth
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "foff");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.foff, sizeof(int));

        // number of if channels
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "nifs");
        outfile.write(field, length * sizeof(char));
        outfile.write((char*)&head.nifs, sizeof(int));

        length = 10;
        outfile.write((char*)&length, sizeof(int));
        strcpy(field, "HEADER_END");
        outfile.write(field, length * sizeof(int));

        size_t to_save = nsamps * head.nchans * head.nbits / 8;

        outfile.write(reinterpret_cast<char*>(odata), to_save);

    }

    outfile.close()


}


#endif
