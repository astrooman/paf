#ifndef _H_PAFRB_GET_MJD
#define _H_PAFRB_GET_MJD

#include <iomanip>

inline double GetMjd(int refepoch, size_t refsecond) {

    // NOTE: Leap year has few rules, but I don't think anyone will be using
    // this code in the year 2100 (sorry if you are), when the divisibility by 100 is taken into account.
    // For now can simply assume, that every year that can be divided by 4 is a leap year.
    short year = 2000 + (int)(refepoch / 2);
    short dinm[12] = {31, 28 , 31, 30 ,31, 30, 31, 31, 30, 31, 30, 31};
    if ((year % 4) == 0)
        dinm[1] = 29;
    short month = 6 * (refepoch % 2);
    short day = (int)(refsecond / 86400);
    int rem = refsecond - 86400 * day;
    day++;
    for (int imonth = 0; imonth < 6; imonth++) {
        if((day - dinm[month + imonth]) > 0) {
            day -= dinm[month];
            month++;
            continue;
        }
        break;
    }
    short hour = (int)(rem / 3600.0);
    short minute = (int)((rem - hour * 3600.0) / 60.0);
    short second = rem - (hour * 3600.0 + minute * 60.0);

    double jd;
    double mjd;
    month++;        // January = 1, February = 2, etc.
    int uttime = hour + ((minute + second/60.0)/60.0);
    jd = 367.0 * year - int(7.0 * (year + (int)((month + 9.0) / 12.0))/ 4.0) + (int)(275.0 * month / 9.0) + day + 1721013.5;
    mjd = jd - 2400000.5;
    double extra = ((double)hour + (((double)minute + (double)second/60.0)/60.0))/ 24.0;
    mjd += extra;

    std::ofstream mydate("mjd.txt", std::ios_base::out | std::ios_base::trunc);
    mydate.precision(6);
    mydate.setf(std::ios::fixed);
    mydate << mjd <<  " " << extra << std::endl;
    mydate.close();

    return mjd;
}

#endif
