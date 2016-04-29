#include <iostream>

using std::cout;
using std::endl;

struct obs_time
{
	int ref_s;
	int start_s;
	int frame_s;
};
void print_struct(obs_time frame)
{
	cout << frame.ref_s << " " << frame.start_s << " " << frame.frame_s << endl;
}

int main(void)
{

	obs_time start{2000,350};
	obs_time some1{2000, 350, 10};
	obs_time some2{2000, 350, 20};

	obs_time *gulp_times = new obs_time[4];

	gulp_times[0] = start;
	gulp_times[1] = some1;
	gulp_times[2] = some2;

	cout << gulp_times->ref_s << " " << (gulp_times + 1)->frame_s << " " << gulp_times[2].frame_s << endl;

	print_struct({start.ref_s, start.start_s, 10});

	return 0;
}
