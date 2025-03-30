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

static void printnpqResults(const vector<pair<vector<short>, vector<vector<float>>>>& results, ostream& os) {
	for (size_t i = 0; i < results.size(); i++) {
		os << "Subspace " << i << ":\n";
		os << "Dimensions: ";
		for (const auto& dim : results[i].first) {
			os << dim << " ";
		}
		os << "\n";
		for (const auto& vec : results[i].second) {
			for (const auto& val : vec) {
				os << val << " ";
			}
			os << "\n";
		}
	}
}

static void printnnpqResults(const vector<pair<vector<short>, int>>& results, ostream& os) {
	for (int i = 0; i < results.size(); i++)
	{
		os << "Subspace " << i << ":\n";
		os << "Dimensions: ";
		for (int j = 0; j < results[i].first.size(); j++)
		{
			os << results[i].first[j] << " ";
		}
		os << "\n";
		os << "Number of codewords: " << results[i].second << "\n";
	}
}

int main()
{
	ifstream file("data_sift_65k.csv");
	string line;
	vector<float> data_f = {};
	bool first = false;
	int n = 0;
	int d = 0;
	while (getline(file, line))
	{
		stringstream ss(line);
		while (getline(ss, line, ','))
		{
			if (!first)
			{
				d++;
			}
			data_f.push_back(stof(line));
		}
		first = true;
		n++;
	}
	cout << "Data size: " << data_f.size() << "\n";
	cout << "Number of vectors: " << n << "\n";
	cout << "Dimensionality: " << d << "\n";
	file.close();

	//auto results = doNaturalProductQuantization(data_f.data(), n, d, 40000, 0.0, -1);
	//printnpqResults(results, cout);
	//ofstream outputs("output_npq.txt");
	//printnpqResults(results, outputs);

	auto result = doNaiveNPQ(data_f.data(), n, d, 40000);
	printnnpqResults(result, cout);
	ofstream output("output_nnpq.txt");
	printnnpqResults(result, output);


	return 0;
}
