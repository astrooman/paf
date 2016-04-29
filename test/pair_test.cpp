#include <iostream>
#include <queue>
#include <utility>
#include <vector>

int main(int argc, char *argv[])
{
	std::queue<std::pair<std::vector<int>,int>> mypair;

	for (int ii = 2; ii < 10; ii++)
		mypair.push(std::pair<std::vector<int>,int>({ii, 2 * ii, 3 * ii}, ii + 1));

	std::vector<int> myvector(3);

	std::copy((mypair.front()).first.begin(), (mypair.front()).first.end(), myvector.begin());

	int myint = (mypair.front()).second;

	std::cout << myint << std::endl;

	std::cout << myvector[0] << " " << myvector[1] << " "  << myvector[2] << std::endl;

	return 0;
}
