#include "natural_product_quantization.h"
#include "algorithm_steps.h"
#include "naive_npq.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

using namespace npq;
using namespace std;

int main()
{
	// open csv file
	ifstream file("data.csv");
	if (!file.is_open())
	{
		cout << "Error opening file" << endl;
		return 1;
	}
	printf("File opened successfully\n");
	// print file contents
	string line;
	while (getline(file, line))
	{
		cout << line << endl;
	}
	file.close();

	return 0;
}

