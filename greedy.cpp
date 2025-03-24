#include <cassert>
#include <numeric>
#include <set>

#include "greedy.h"


namespace npq
{

double subspaceCost(double expEntropy, dim_t subspaceDims, id_t numVectors)
{
	const double scalarSize = sizeof(scalar_t) * 8;
	return numVectors * log2(expEntropy) + expEntropy * subspaceDims * scalarSize;
}


bool MergeOperation::operator<(const MergeOperation& other) const
{
	return changeInTotalCost < other.changeInTotalCost;
}

MergeOperation::MergeOperation(double changeInTotalCost, const std::pair<dim_t, dim_t>& edge, Partition&& newPartition, double newCost)
	: changeInTotalCost(changeInTotalCost),
	edge(edge),
	newPartition(std::move(newPartition)),
	newCost(newCost)
{}

MergeOperation::MergeOperation(MergeOperation&& other) noexcept
	: changeInTotalCost(other.changeInTotalCost),
	edge(std::move(other.edge)),
	newPartition(std::move(other.newPartition)),
	newCost(other.newCost)
{}

MergeOperation& MergeOperation::operator=(MergeOperation&& other) noexcept
{
	changeInTotalCost = other.changeInTotalCost;
	edge = std::move(other.edge);
	newPartition = std::move(other.newPartition);
	newCost = other.newCost;

	return *this;
}

GreedyMerger::GreedyMerger(SubspaceDecomposition& decomp, const Graph& tree, const std::vector<Partition>& partitions)
	: pDecomp(&decomp), pTree(&tree), pPartitions(&partitions)
{
	const id_t numVectors = partitions[0].vecIdToBlockId.size();

	// Compute the joint partition and cost of each subspace
	for (const dim_t& subspaceId : decomp.allSubspaceIds())
	{
		const std::vector<dim_t> dimIds = decomp.getDimIds(subspaceId);
		subspaceIdToJointPartition[subspaceId] = jointPartitionByIndices(partitions, dimIds);

		const double subspaceExpEntropy = entropy(subspaceIdToJointPartition[subspaceId], true);
		subspaceIdToCost[subspaceId] = subspaceCost(subspaceExpEntropy, dimIds.size(), numVectors);
	}

	// Populate the merge queue with all possible merges, corresponding to edges between different subspaces
    for (const auto& edge : tree.edges)
    {
        const dim_t dimId1 = edge.first;
        const dim_t dimId2 = edge.second;
		if (decomp.inSameSubspace(dimId1, dimId2))
		{
			continue;
		}

		const dim_t subspaceId1 = decomp.getSubspaceId(dimId1);
		const dim_t subspaceId2 = decomp.getSubspaceId(dimId2);
		const std::vector<dim_t> dimIds1 = decomp.getDimIds(subspaceId1);
		const std::vector<dim_t> dimIds2 = decomp.getDimIds(subspaceId2);

		Partition newPartition = jointPartition(subspaceIdToJointPartition[subspaceId1],
												subspaceIdToJointPartition[subspaceId2]);

		const double newExpEntropy = entropy(newPartition, true);
		const double newCost = subspaceCost(newExpEntropy, dimIds1.size() + dimIds2.size(), numVectors);
		const double changeInTotalCost = newCost - subspaceIdToCost[subspaceId1] - subspaceIdToCost[subspaceId2];

		// TODO: For now, we enqueue a merge even if the cost increases, to make updating the queue easier later on
		mergeQueue.emplace_back(MergeOperation{ changeInTotalCost, edge, std::move(newPartition), newCost });
    }

	// Set loss of 0-th iteration to the total cost of the decomposition
	losses.push_back(std::accumulate(
		subspaceIdToCost.begin(),
		subspaceIdToCost.end(),
		0.0,
		[](double sum, const auto& pair) { return sum + pair.second; }
	));
}

bool GreedyMerger::canMerge() const
{
	if (mergeQueue.empty())
	{
		return false;
	}

	const MergeOperation& bestMerge = *std::min_element(mergeQueue.begin(), mergeQueue.end());
	return bestMerge.changeInTotalCost < 0.0;
}

void GreedyMerger::doMerge()
{
	// Get the best merge operation from the queue
	assert(canMerge());
	std::vector<MergeOperation>::iterator bestMergeIt = std::min_element(mergeQueue.begin(), mergeQueue.end());
	MergeOperation bestMerge = std::move(*bestMergeIt);
	mergeQueue.erase(bestMergeIt);

	// Look up the subspaces involved in the merge
	const dim_t dimId1 = bestMerge.edge.first;
	const dim_t dimId2 = bestMerge.edge.second;
	const dim_t subspaceId1 = pDecomp->getSubspaceId(dimId1);
	const dim_t subspaceId2 = pDecomp->getSubspaceId(dimId2);
	assert(subspaceId1 != subspaceId2);

	// Save this information before merging for an assertion after
	const std::vector<dim_t> dimIds1 = pDecomp->getDimIds(subspaceId1);
	const std::vector<dim_t> dimIds2 = pDecomp->getDimIds(subspaceId2);

	// Clean up the old subspaces from the maps
	subspaceIdToJointPartition.erase(subspaceId1);
	subspaceIdToJointPartition.erase(subspaceId2);
	subspaceIdToCost.erase(subspaceId1);
	subspaceIdToCost.erase(subspaceId2);

	// Do the merge
	pDecomp->merge(dimId1, dimId2);
	assert(pDecomp->isValidSplit(dimIds1, dimIds2));
	const dim_t newSubspaceId = pDecomp->getSubspaceId(dimId1);
	assert(newSubspaceId == pDecomp->getSubspaceId(dimId2));

	// Add the new subspace to the maps
	subspaceIdToJointPartition[newSubspaceId] = std::move(bestMerge.newPartition);
	subspaceIdToCost[newSubspaceId] = bestMerge.newCost;

	// Update the invalidated merges in the queue
	const id_t numVectors = pPartitions->front().vecIdToBlockId.size();
	for (MergeOperation& op : mergeQueue)
	{
		// Note: Variables subspaceId1 and subspaceId2 in the outer scope are shadowed
		const dim_t subspaceId1 = pDecomp->getSubspaceId(op.edge.first);
		const dim_t subspaceId2 = pDecomp->getSubspaceId(op.edge.second);
		assert(subspaceId1 != subspaceId2);
		if (subspaceId1 != newSubspaceId && subspaceId2 != newSubspaceId)
		{
			// This merge does not involve the new subspace, so it doesn't need to be updated
			continue;
		}

		// Update the merge operation in the exact same way it was computed in the constructor
		const std::vector<dim_t> dimIds1 = pDecomp->getDimIds(subspaceId1);
		const std::vector<dim_t> dimIds2 = pDecomp->getDimIds(subspaceId2);

		Partition newPartition = jointPartition(subspaceIdToJointPartition[subspaceId1],
												subspaceIdToJointPartition[subspaceId2]);

		const double newExpEntropy = entropy(newPartition, true);
		const double newCost = subspaceCost(newExpEntropy, dimIds1.size() + dimIds2.size(), numVectors);
		const double changeInTotalCost = newCost - subspaceIdToCost[subspaceId1] - subspaceIdToCost[subspaceId2];

		op.changeInTotalCost = changeInTotalCost;
		// op.edge is unchanged
		op.newPartition = std::move(newPartition);
		op.newCost = newCost;
	}

	// Record the loss of the current iteration
	losses.push_back(losses.back() + bestMerge.changeInTotalCost);
}

std::vector<double> GreedyMerger::getLosses() const
{
	return losses;
}

} // namespace npq