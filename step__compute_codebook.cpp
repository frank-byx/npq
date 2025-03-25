/**
 * @brief The fourth step of the algorithm. See algorithm_steps.h for more information.
 */

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
			Subspace dimIds = decomp.getDimIds(subspaceId);
			Partition subspaceJointPartition = jointPartitionByIndices(partitions, dimIds);

			std::vector<Codeword> codewords;
			codewords.reserve(subspaceJointPartition.blockIdToVecIds.size());
			for (const std::vector<id_t>& cluster : subspaceJointPartition.blockIdToVecIds)
			{
				Codeword codeword(dimIds.size(), 0.0f);

				for (const id_t& vecId : cluster)
				{
					for (dim_t i = 0; i < dimIds.size(); ++i)
					{
						codeword[i] += dataset.dimensions[dimIds[i]][vecId];
					}
				}

				for (dim_t i = 0; i < dimIds.size(); ++i)
				{
					codeword[i] /= cluster.size();
				}

				codewords.push_back(std::move(codeword));
			}

			codebook.emplace_back(std::move(dimIds), std::move(codewords));
		}

		return codebook;
	}

} // namespace npq