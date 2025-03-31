#include <cassert>
#include <iostream>
#include <map>
#include <numeric>

#include "compute_subspace_decomp_naive.h"


namespace npq
{

// TODO: This function is copied from greedy.cpp
double _subspaceCost(double expEntropy, dim_t subspaceDims, id_t trueNumVectors)
{
	const double scalarSize = sizeof(scalar_t) * 8;
	return trueNumVectors * log2(expEntropy) + expEntropy * subspaceDims * scalarSize;
}

// Merge operation (changeInTotalCost, (subspaceId1, subspaceId2)), where subspaceId1 < subspaceId2
using MergeOp = std::pair<double, std::pair<dim_t, dim_t>>;

SubspaceDecomposition computeSubspaceDecompositionNaive(const std::vector<Partition>& partitions, const Parameters& params)
{
	const dim_t numDims = partitions.size();

	// Initial subspace decomposition of singleton dimensions
	std::vector<dim_t> dimIdToSubspaceId;
	dimIdToSubspaceId.resize(numDims);
	std::iota(dimIdToSubspaceId.begin(), dimIdToSubspaceId.end(), 0);

	// Set the maps for the initial subspace decomposition
	std::map<dim_t, Partition> subspaceIdToJointPartition;
	std::map<dim_t, double> subspaceIdToCost;
	for (dim_t subspaceId = 0; subspaceId < numDims; ++subspaceId)
	{
		subspaceIdToJointPartition[subspaceId] = partitions[subspaceId].copy();
		const double subspaceExpEntropy = entropy(subspaceIdToJointPartition[subspaceId], true);
		subspaceIdToCost[subspaceId] = _subspaceCost(subspaceExpEntropy, 1, params.trueNumVectors);
	}

	// Populate the merge queue with all possible merges of pairs of dimensions
	std::vector<MergeOp> mergeQueue;
	mergeQueue.reserve(numDims * (numDims - 1) / 2);
	for (dim_t subspaceId1 = 0; subspaceId1 < numDims; ++subspaceId1)
	{
		for (dim_t subspaceId2 = subspaceId1 + 1; subspaceId2 < numDims; ++subspaceId2)
		{
			const double newExpEntropy = entropy(jointPartition(subspaceIdToJointPartition[subspaceId1],
				subspaceIdToJointPartition[subspaceId2]), true);
			const double newCost = _subspaceCost(newExpEntropy, 2, params.trueNumVectors);
			const double changeInTotalCost = newCost - subspaceIdToCost[subspaceId1] - subspaceIdToCost[subspaceId2];

			mergeQueue.emplace_back(MergeOp{ changeInTotalCost, { subspaceId1, subspaceId2 } });
		}
	}

	// Record the initial loss as the total cost of the initial decomposition
	double loss = std::accumulate(subspaceIdToCost.begin(), subspaceIdToCost.end(), 0.0,
		[](double acc, const std::pair<dim_t, double>& p) { return acc + p.second; });
	std::cout << "Cost of initial subspace decomposition: " << loss << std::endl;

	// Do the greedy merging
	int mergeCount = 0;
	while (!mergeQueue.empty())
	{
		// Get the best merge operation from the queue
		std::vector<MergeOp>::iterator bestMergeIt = std::min_element(mergeQueue.begin(), mergeQueue.end());
		MergeOp bestMerge = std::move(*bestMergeIt);
		mergeQueue.erase(bestMergeIt);
		if (bestMerge.first >= 0.0)
		{
			// No more merges that decrease the total cost of the decomposition
			break;
		}

		const dim_t subspaceId1 = bestMerge.second.first;
		const dim_t subspaceId2 = bestMerge.second.second;
		assert(subspaceId1 < subspaceId2);
		Partition partition1 = std::move(subspaceIdToJointPartition[subspaceId1]);
		Partition partition2 = std::move(subspaceIdToJointPartition[subspaceId2]);

		// Clean up the old subspaces from the maps
		subspaceIdToJointPartition.erase(subspaceId1);
		subspaceIdToJointPartition.erase(subspaceId2);
		subspaceIdToCost.erase(subspaceId1);
		subspaceIdToCost.erase(subspaceId2);

		// Do the merge, setting the first (lower) subspace ID as the new subspace ID
		for (dim_t i = 0; i < numDims; ++i)
		{
			if (dimIdToSubspaceId[i] == subspaceId2)
			{
				dimIdToSubspaceId[i] = subspaceId1;
			}
		}

		// Add the new subspace to the maps
		subspaceIdToJointPartition[subspaceId1] = jointPartition(partition1, partition2);
		const double newExpEntropy = entropy(subspaceIdToJointPartition[subspaceId1], true);
		dim_t newSubspaceDims = 0;
		for (dim_t dimId = 0; dimId < numDims; ++dimId)
		{
			if (dimIdToSubspaceId[dimId] == subspaceId1)
			{
				++newSubspaceDims;
			}
		}
		subspaceIdToCost[subspaceId1] = _subspaceCost(newExpEntropy, newSubspaceDims, params.trueNumVectors);

		// Remove merges in the queue that involve the second merged subspace
		mergeQueue.erase(
			std::remove_if(
				mergeQueue.begin(),
				mergeQueue.end(),
				[&subspaceId2](const MergeOp& op) {
					return op.second.first == subspaceId2 || op.second.second == subspaceId2;
				}
			),
			mergeQueue.end()
		);

		// Update the merges in the queue that involve the first merged subspace
		for (MergeOp& op : mergeQueue)
		{
			if (op.second.first != subspaceId1 && op.second.second != subspaceId1)
			{
				// This merge does not involve the new subspace, so it doesn't need to be updated
				continue;
			}

			const double opExpEntropy = entropy(jointPartition(subspaceIdToJointPartition[op.second.first],
				subspaceIdToJointPartition[op.second.second]), true);
			dim_t opNumDims = 0;
			for (dim_t dimId = 0; dimId < numDims; ++dimId)
			{
				if (dimIdToSubspaceId[dimId] == op.second.first || dimIdToSubspaceId[dimId] == op.second.second)
				{
					++opNumDims;
				}
			}
			const double opCost = _subspaceCost(opExpEntropy, opNumDims, params.trueNumVectors);
			op.first = opCost - subspaceIdToCost[op.second.first] - subspaceIdToCost[op.second.second];
		}

		// Record the loss of the current iteration
		loss = loss + bestMerge.first;
		++mergeCount;
		std::cout << "Cost after " << mergeCount << " merge(s): " << loss << std::endl;
	}

	return SubspaceDecomposition{ std::move(dimIdToSubspaceId) };
}

} // namespace npq