#include <bitset>
#include <iostream>
#include <vector>


using std::cout;
using std::endl;
using std::vector;

#define DATA 16

int main(int argc, char *argv[])
{

    unsigned char *h_data = new unsigned char[DATA];

    for (int ii = 0; ii < DATA; ii++)
        h_data[ii] = (unsigned char)ii * 2 + 1;

    cout << "Original data:" << endl;
    for (int ii = 0; ii < DATA; ii++)
        cout << (int)h_data[ii] << endl;


    int offset = DATA;

    vector<unsigned char> d_data_c(h_data, h_data + offset);
    vector<unsigned short> d_data_s(reinterpret_cast<unsigned short*>(h_data), reinterpret_cast<unsigned short*>(h_data + offset));
    vector<float> d_data_f(reinterpret_cast<float*>(h_data), reinterpret_cast<float*>(h_data + offset));

    float *my_float = new float[1];
    unsigned char *my_char = new unsigned char[4];

    my_float[0] = 3.1415926;

    my_char = reinterpret_cast<unsigned char*>(my_float);

    float *my_float2 = new float[1];
    my_float2 = reinterpret_cast<float*>(my_char);

    cout << std::bitset<8>(my_char[3]) << std::bitset<8>(my_char[2]) << std::bitset<8>(my_char[1]) << std::bitset<8>(my_char[0]) << endl;

    cout << my_float[0] << ", " << my_float2[0] << endl;

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
