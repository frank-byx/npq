#include <cassert>
#include <cmath>

#include "correction.h"


namespace npq
{

double estimateNumClusters(double expEntropy, const std::vector<dim_t>& subspaceDimIds, const Parameters& params, const std::vector<Partition>& partitions)
{
	if (!params.useCorrection)
	{
		return expEntropy;
	}

	const id_t subsampleSize = partitions[0].vecIdToBlockId.size();
	const id_t fullDatasetSize = params.trueNumVectors;
	double logRatio = log2(subsampleSize) / log2(fullDatasetSize);
	assert(logRatio > 0.0 && logRatio <= 1.0);

	double kProd = 1.0;
	for (const dim_t& dimId : subspaceDimIds)
	{
		const id_t k = partitions[dimId].blockIdToVecIds.size();
		kProd *= k;
	}
	assert(expEntropy <= kProd);

	const double correctedNumClusters = logRatio * expEntropy + (1.0 - logRatio) * kProd;
	return correctedNumClusters;
}

} // namespace npq
