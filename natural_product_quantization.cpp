#include "natural_product_quantization.h"
#include "algorithm_steps.h"
#include "naive_npq.h"


using namespace npq;

Codebook _doNaturalProductQuantization(const scalar_t* data, id_t n, dim_t d,
									   double targetDistortion, double targetDistortionMargin, dim_t mstMaxDegree)
{
	const Dataset dataset{ data, n, d };
	const Parameters params{ targetDistortion, targetDistortionMargin, mstMaxDegree };
	
	// Step 1: Compute 1D partitions
	const std::vector<Partition> partitions = compute1dPartitions(dataset, params);

	// Step 2: Compute constraint tree
	const Graph tree = computeVIMST(partitions, params);

	// Step 3: Compute subspace decomposition
	const SubspaceDecomposition subspaceDecomposition = computeSubspaceDecomposition(partitions, tree, params);

	// Step 4: Compute codebook
	return computeCodebook(subspaceDecomposition, partitions, dataset, params);
}

std::vector<std::pair<std::vector<short>, std::vector<std::vector<float>>>>
doNaturalProductQuantization(const float* data, int n, short d, double targetDistortion, double targetDistortionMargin, short mstMaxDegree)
{
	return _doNaturalProductQuantization(data, n, d, targetDistortion, targetDistortionMargin, mstMaxDegree);
}

std::vector<std::pair<std::vector<short>, int>> doNaiveNPQ(const float* data, int n, short d, double targetDistortion)
{
	const Dataset dataset{ data, n, d };

	return nnpq::naiveNPQ(dataset, targetDistortion);
}
