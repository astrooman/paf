#include <bitset>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    unsigned char header[8];
    bool flag;

    for (int ii = 0; ii < 8; ii++)
        header[ii] = ii + 1;

    for (int ii = 0; ii < 8; ii++)
        cout << (int)header[ii] << " -> " << std::bitset<8>(header[ii]) << endl;

    int try1 = header[0] | (header[1] << 8) | (header[2] << 16) | (header[3] << 24);

    flag = (bool)(header[1] >> 0);

    cout << flag << endl;

    long sipp = (long)(header[0] | (header[1] << 8) | (header[2] << 16) | (header[3] << 24)
                    | ((long)header[4] << 32) | ((long)header[5] << 40) | ((long)header[6] << 48) | ((long)header[7] << 56));

    cout << (int)(header[1] << 8) << endl;
    cout << try1 << " -> " << std::bitset<32>(try1) << endl;
    cout << std::bitset<16>(header[1]) << endl;
    cout << sipp << " -> " << std::bitset<64>(sipp) << endl;

    cout << sizeof(long);
    return 0;
}
