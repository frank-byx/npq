/**
 * @brief The fourth step of the algorithm. See algorithm_steps.h for more information.
 */

#include <algorithm>

#include "algorithm_steps.h"
#include "subspace_decomposition.h"


namespace npq
{

Codebook computeCodebook(const SubspaceDecomposition& decomp,
						 const std::vector<Partition>& partitions,
						 const Dataset& dataset,
						 const Parameters& params)
{
	Codebook codebook;
	const std::vector<dim_t> subspaceIds = decomp.allSubspaceIds();
	codebook.reserve(subspaceIds.size());
	for (const dim_t& subspaceId : subspaceIds)
	{
		// For each subspace, the number of codewords is the exponential entropy of the joint partition, rounded up to
		// the nearest power of 2 to make full use of the encoding bits, but capped by the number of partition blocks
		Subspace dimIds = decomp.getDimIds(subspaceId);
		Partition subspaceJointPartition = jointPartitionByIndices(partitions, dimIds);
		id_t numCodewords = 1 << static_cast<int>(ceil(entropy(subspaceJointPartition, false)));
		if (numCodewords > subspaceJointPartition.blockIdToVecIds.size())
		{
			numCodewords = static_cast<id_t>(subspaceJointPartition.blockIdToVecIds.size());
		}

		// We use the largest partition blocks as the clusters to initialize the codewords
		std::vector<std::vector<id_t>> clusters = std::move(subspaceJointPartition.blockIdToVecIds);
		std::sort(clusters.begin(), clusters.end(),
				  [](const std::vector<id_t>& a, const std::vector<id_t>& b) { return a.size() > b.size(); });

		std::vector<Codeword> codewords;
		codewords.reserve(numCodewords);
		for (id_t i = 0; i < numCodewords; ++i)
		{
			Codeword codeword(dimIds.size(), 0.0f);

			for (const id_t& vecId : clusters[i])
			{
				for (dim_t i = 0; i < dimIds.size(); ++i)
				{
					codeword[i] += dataset.dimensions[dimIds[i]][vecId];
				}
			}

			for (dim_t i = 0; i < dimIds.size(); ++i)
			{
				codeword[i] /= clusters[i].size();
			}

			codewords.push_back(std::move(codeword));
		}

		codebook.emplace_back(std::move(dimIds), std::move(codewords));
	}

	return codebook;
}

} // namespace npq