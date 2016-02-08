#include <iostream>
#include <thrust/device_vector.h>

using std::cout;
using std::endl;
using thrust::device_vector;

#define DATA 16

int main(int argc, char *argv[])
{

    unsigned char *h_data = new unsigned char[DATA];

    for (int ii = 0; ii < DATA; ii++)
        h_data = (unsigned char)ii * 2 + 1;

    cout << "Original data:" << endl;
    for (int ii = 0; ii < DATA; ii++)
        cout << (int)h_data[ii] << endl;


    device_vector d_data_c((unsigned char*)h_data, (unsigned char*)h_data + offset);
    device_vector d_data_s((unsigned short*)h_data, (unsigned short*)h_data + offset);
    device_vector d_data_f((float*)h_data, (float*)h_data + offset);

    cout << "Char device:" << endl;
    for (int ii = 0; ii < DATA; ii++)
        cout << (int)d_data_c[ii] << endl;

    cout << "Short device:" << endl;
    for (int ii = 0; ii < DATA; ii++)
        cout << d_data_s[ii] << endl;

    cout << "Float device:" << endl;
    for (int ii = 0; ii < DATA; ii++)
        cout << d_data_f[ii] << endl;

    return 0;
}
