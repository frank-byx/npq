#include "natural_product_quantization.h"
#include "algorithm_steps.h"
#include "naive_npq.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <sstream>

using namespace std;
using namespace filesystem;

int main()
{
	ifstream file("data3.csv");
	string line;
	getline(file, line);
	stringstream ss (line);
	vector<string> data_s = {};
	while (getline(ss, line, ','))
	{
		data_s.push_back(line);
	}
	vector<float> data_f = {};
	// convert string to float
	for (int i = 0; i < data_s.size(); i++)
	{
		data_f.push_back(stof(data_s[i]));
	}
	file.close();

	auto results = doNaturalProductQuantization(data_f.data(), 100, 10, 500, 0.0, -1);

	for (int i = 0; i < results.size(); i++)
	{
		cout << "Subspace " << i << ":\n";
		cout << "Dimensions: ";
		for (int j = 0; j < results[i].first.size(); j++)
		{
			cout << results[i].first[j] << " ";
		}
		cout << "\n";
		for (int j = 0; j < results[i].second.size(); j++)
		{
			for (int k = 0; k < results[i].second[j].size(); k++)
			{
				cout << results[i].second[j][k] << " ";
			}
			cout << "\n";
		}
	}

	// write to text file

	ofstream output("output.txt");
	for (int i = 0; i < results.size(); i++)
	{
		output << "Subspace " << i << ":\n";
		output << "Dimensions: ";
		for (int j = 0; j < results[i].first.size(); j++)
		{
			output << results[i].first[j] << " ";
		}
		output << "\n";
		for (int j = 0; j < results[i].second.size(); j++)
		{
			for (int k = 0; k < results[i].second[j].size(); k++)
			{
				output << results[i].second[j][k] << " ";
			}
			cout << "\n";
		}
	}

	auto result = doNaiveNPQ(data_f.data(), 100, 10, 500);

	for (int i = 0; i < result.size(); i++)
	{
		cout << "Subspace " << i << ":\n";
		cout << "Dimensions: ";
		for (int j = 0; j < result[i].first.size(); j++)
		{
			cout << result[i].first[j] << " ";
		}
		cout << "\n";
		cout << "Number of codewords: " << result[i].second << "\n";
	}

	// write to text file
	ofstream output2("output2.txt");

	for (int i = 0; i < result.size(); i++)
	{
		output2 << "Subspace " << i << ":\n";
		output2 << "Dimensions: ";
		for (int j = 0; j < result[i].first.size(); j++)
		{
			output2 << result[i].first[j] << " ";
		}
		output2 << "\n";
		output2 << "Number of codewords: " << result[i].second << "\n";
	}

	return 0;
}
