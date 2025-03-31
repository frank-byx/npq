#include "natural_product_quantization.h"
#include "algorithm_steps.h"
#include "naive_npq.h"
#include "compute_subspace_decomp_naive.h"

#include <iostream>


using namespace npq;

Codebook _doNaturalProductQuantization(const scalar_t* data, id_t n, dim_t d,
									   id_t trueNumVectors, double targetDistortion, double targetDistortionMargin, dim_t mstMaxDegree)
{
	const Dataset dataset{ data, n, d };
	const Parameters params{ trueNumVectors, targetDistortion, targetDistortionMargin, mstMaxDegree };
	
	// Step 1: Compute 1D partitions
	std::cout << "Step 1: Compute 1D partitions" << std::endl;
	const std::vector<Partition> partitions = compute1dPartitions(dataset, params);

	// Step 2: Compute constraint tree
	std::cout << "Step 2: Compute constraint tree" << std::endl;
	const Graph tree = computeVIMST(partitions, params);

	// Step 3: Compute subspace decomposition
	std::cout << "Step 3: Compute subspace decomposition" << std::endl;
	const SubspaceDecomposition subspaceDecomposition = computeSubspaceDecomposition(partitions, tree, params);

	// Step 4: Compute codebook
	std::cout << "Step 4: Compute codebook" << std::endl;
	return computeCodebook(subspaceDecomposition, partitions, dataset, params);
}

std::vector<std::pair<std::vector<short>, std::vector<std::vector<float>>>>
doNaturalProductQuantization(const float* data, int n, short d, int trueNumVectors, double targetDistortion, double targetDistortionMargin, short mstMaxDegree)
{
	return _doNaturalProductQuantization(data, n, d, trueNumVectors, targetDistortion, targetDistortionMargin, mstMaxDegree);
}

std::vector<std::pair<std::vector<short>, int>> doNaiveNPQ(const float* data, int n, short d, int trueNumVectors, double targetDistortion)
{
	const Dataset dataset{ data, n, d };

	return nnpq::naiveNPQ(dataset, trueNumVectors, targetDistortion);
}


Codebook _doNotSoNaiveNPQ(const scalar_t* data, id_t n, dim_t d,
						  id_t trueNumVectors, double targetDistortion, double targetDistortionMargin)
{
	std::cout << "Running doNotSoNaiveNPQ." << std::endl;

	const Dataset dataset{ data, n, d };
	const Parameters params{ trueNumVectors, targetDistortion, targetDistortionMargin };

	// Step 1: Compute 1D partitions
	std::cout << "Step 1: Compute 1D partitions" << std::endl;
	const std::vector<Partition> partitions = compute1dPartitions(dataset, params);

	// Step 3: Compute subspace decomposition using naive greedy merging
	std::cout << "Step 3: Compute subspace decomposition " << std::endl;
	const SubspaceDecomposition subspaceDecomposition = computeSubspaceDecompositionNaive(partitions, params);

	// Step 4: Compute codebook
	std::cout << "Step 4: Compute codebook" << std::endl;
	return computeCodebook(subspaceDecomposition, partitions, dataset, params);
}

std::vector<std::pair<std::vector<short>, std::vector<std::vector<float>>>> doNotSoNaiveNPQ(const float* data, int n, short d, int trueNumVectors, double targetDistortion, double targetDistortionMargin)
{
	return _doNotSoNaiveNPQ(data, n, d, trueNumVectors, targetDistortion, targetDistortionMargin);
}
