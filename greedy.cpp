#include <cassert>
#include <map>
#include <numeric>
#include <set>

#include "greedy.h"


namespace npq
{

double subspaceCost(double expEntropy, dim_t subspaceDims, id_t trueNumVectors)
{
	const double scalarSize = sizeof(scalar_t) * 8;
	return trueNumVectors * log2(expEntropy) + expEntropy * subspaceDims * scalarSize;
}


bool MergeOperation::operator<(const MergeOperation& other) const
{
	return changeInTotalCost < other.changeInTotalCost;
}

MergeOperation::MergeOperation(double changeInTotalCost,
							   const std::pair<dim_t, dim_t>& edge,
							   Partition&& newPartition,
							   double newCost)
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

GreedyMerger::GreedyMerger(SubspaceDecomposition& decomp, const Graph& tree, const std::vector<Partition>& partitions, const Parameters& params)
	: pDecomp(&decomp), pTree(&tree), pPartitions(&partitions), trueNumVectors(params.trueNumVectors)
{
	const id_t numVectors = partitions[0].vecIdToBlockId.size();

	// Compute the joint partition and cost of each subspace
	for (const dim_t& subspaceId : decomp.allSubspaceIds())
	{
		const std::vector<dim_t> dimIds = decomp.getDimIds(subspaceId);
		subspaceIdToJointPartition[subspaceId] = jointPartitionByIndices(partitions, dimIds);

		const double subspaceExpEntropy = entropy(subspaceIdToJointPartition[subspaceId], true);
		subspaceIdToCost[subspaceId] = subspaceCost(subspaceExpEntropy, dimIds.size(), trueNumVectors);
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
		const double newCost = subspaceCost(newExpEntropy, dimIds1.size() + dimIds2.size(), trueNumVectors);
		const double changeInTotalCost = newCost - subspaceIdToCost[subspaceId1] - subspaceIdToCost[subspaceId2];

		// TODO: For now, we enqueue a merge even if the cost increases, to make updating the queue easier later on
		mergeQueue.emplace_back(MergeOperation{ changeInTotalCost, edge, std::move(newPartition), newCost });
    }

	// Set loss of 0-th iteration to the total cost of the decomposition
	losses.push_back(std::accumulate(subspaceIdToCost.begin(), subspaceIdToCost.end(),
									 0.0, [](double sum, const auto& pair) { return sum + pair.second; }));
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
		const double newCost = subspaceCost(newExpEntropy, dimIds1.size() + dimIds2.size(), trueNumVectors);
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


bool SplitOperation::operator<(const SplitOperation& other) const
{
	return changeInTotalCost < other.changeInTotalCost;
}

SplitOperation::SplitOperation(double changeInTotalCost,
							   const std::pair<dim_t, dim_t>& edge,
							   const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2,
							   double newCost1, double newCost2)
	: changeInTotalCost(changeInTotalCost),
	  edge(edge),
	  dimIds1(dimIds1), dimIds2(dimIds2),
	  newCost1(newCost1), newCost2(newCost2)
{}

SplitOperation::SplitOperation(SplitOperation && other) noexcept
    : changeInTotalCost(other.changeInTotalCost),
	  edge(std::move(other.edge)),
	  dimIds1(std::move(other.dimIds1)), dimIds2(std::move(other.dimIds2)),
	  newCost1(other.newCost1), newCost2(other.newCost2)
{}

SplitOperation& SplitOperation::operator=(SplitOperation && other) noexcept
{
	changeInTotalCost = other.changeInTotalCost;
	edge = std::move(other.edge);
	dimIds1 = std::move(other.dimIds1);
	dimIds2 = std::move(other.dimIds2);
	newCost1 = other.newCost1;
	newCost2 = other.newCost2;

	return *this;
}

void GreedySplitter::postorderDFS(dim_t curDimId, dim_t parentDimId,
								  std::map<dim_t, Partition>& subtreeJointPartitions,
								  std::map<dim_t, std::vector<dim_t>>& subtreeDimIds,
								  std::map<dim_t, dim_t>& childToParentDimId)
{
	if (parentDimId != -1)
	{
		childToParentDimId[curDimId] = parentDimId;
	}

	Partition curJointPartition{ (*pPartitions)[curDimId].copy() };
	std::vector<dim_t> curDimIds = { curDimId };

	for (const dim_t& childDimId : pTree->adj[curDimId])
	{
		if (childDimId == parentDimId || !pDecomp->inSameSubspace(childDimId, curDimId))
		{
			continue;
		}

		postorderDFS(childDimId, curDimId, subtreeJointPartitions, subtreeDimIds, childToParentDimId);

		curJointPartition = jointPartition(curJointPartition, subtreeJointPartitions[childDimId]);
		for (const dim_t& dimId : subtreeDimIds[childDimId])
		{
			curDimIds.push_back(dimId);
		}
	}

	subtreeJointPartitions[curDimId] = std::move(curJointPartition);
	subtreeDimIds[curDimId] = std::move(curDimIds);
}

void GreedySplitter::preorderDFS(dim_t curDimId, dim_t parentDimId,
								 std::map<dim_t, Partition>& complementJointPartitions,
								 std::map<dim_t, std::vector<dim_t>>& complementDimIds,
								 const std::map<dim_t, Partition>& subtreeJointPartitions,
								 const std::map<dim_t, std::vector<dim_t>>& subtreeDimIds)
{
	// The parent will have already computed the complement of the subtree at the current node,
	// and we will be computing the complement for our children.

	const id_t numVectors = pPartitions->front().vecIdToBlockId.size();

	// Compute the joint partition of the common subset of the complements of the children,
	// which is just the current node plus the complement of the subtree at the current node.
	Partition curJointPartition{ (*pPartitions)[curDimId].copy() };
	std::vector<dim_t> curDimIds = { curDimId };
	if (parentDimId != -1)
	{
		curJointPartition = jointPartition(curJointPartition, complementJointPartitions[curDimId]);
		for (const dim_t& dimId : complementDimIds[curDimId])
		{
			curDimIds.push_back(dimId);
		}
	}

	for (const dim_t& childDimId : pTree->adj[curDimId])
	{
		if (childDimId == parentDimId || !pDecomp->inSameSubspace(childDimId, curDimId))
		{
			continue;
		}

		Partition childComplementPartition{ curJointPartition.copy() };
		std::vector<dim_t> childComplementDimIds = curDimIds;

		// Accumulate all sibling subtrees into the complement of the current child
		for (const dim_t& otherChildDimId : pTree->adj[curDimId])
		{
			if (otherChildDimId == childDimId
				|| otherChildDimId == parentDimId || !pDecomp->inSameSubspace(otherChildDimId, curDimId))
			{
				continue;
			}

			childComplementPartition = jointPartition(childComplementPartition, subtreeJointPartitions.at(otherChildDimId));
			for (const dim_t& dimId : subtreeDimIds.at(otherChildDimId))
			{
				childComplementDimIds.push_back(dimId);
			}
		}

		complementJointPartitions[childDimId] = std::move(childComplementPartition);
		complementDimIds[childDimId] = std::move(childComplementDimIds);

		preorderDFS(childDimId, curDimId,
					complementJointPartitions, complementDimIds,
					subtreeJointPartitions, subtreeDimIds);
	}
}

bool compareSets(const std::vector<dim_t>& a, const std::vector<dim_t>& b)
{
	auto a_set = std::set<dim_t>(a.begin(), a.end());
	auto b_set = std::set<dim_t>(b.begin(),b.end());

	return a_set == b_set;
}

void GreedySplitter::enqueueSubspace(dim_t subspaceId)
{
	const id_t numVectors = pPartitions->front().vecIdToBlockId.size();

	// Root the subgraph of the tree induced by the current subspace at any node/dimension,
	// WLOG choosing the dimension that has the same ID as the subspace for convenience.
	const dim_t rootDimId = subspaceId;

	// Observe that each split of the subspace corresponds to removing one edge of the rooted tree,
	// leaving two connected components: a rooted subtree, and the complement of the rooted subtree.

	// First, do a postorder traversal of the rooted tree to compute the joint partition of each rooted subtree.
	std::map<dim_t, Partition> subtreeJointPartitions;
	std::map<dim_t, std::vector<dim_t>> subtreeDimIds;  // The dimensions in each subtree
	std::map<dim_t, dim_t> childToParentDimId;  // Keep track of the parents to restore the edges later
	postorderDFS(rootDimId, -1, subtreeJointPartitions, subtreeDimIds, childToParentDimId);
	
	assert(compareSets(subtreeDimIds.at(rootDimId), pDecomp->getDimIds(subspaceId)));
	assert(subtreeJointPartitions.at(rootDimId) == jointPartitionByIndices(*pPartitions, pDecomp->getDimIds(subspaceId)));

	// Then, do a preorder traversal of the rooted tree to compute the joint partitions of the complements.
	// We use the postorder traversal results to avoid recomputing the joint partitions of the subtrees.
	std::map<dim_t, Partition> complementJointPartitions;
	std::map<dim_t, std::vector<dim_t>> complementDimIds;
	preorderDFS(rootDimId, -1, complementJointPartitions, complementDimIds, subtreeJointPartitions, subtreeDimIds);

	// Compute the cost of the subspace
	// TODO: Avoid recomputing the cost of the subspace after the first time this function is called in the constructor
	const double expEntropy = entropy(subtreeJointPartitions.at(rootDimId), true);
	subspaceIdToCost[subspaceId] = subspaceCost(expEntropy, subtreeDimIds.at(rootDimId).size(), trueNumVectors);

	// Populate the split queue with each pair of rooted subtree and complement
	for (const auto& [childDimId, parentDimId] : childToParentDimId)
	{
		assert(pDecomp->inSameSubspace(childDimId, parentDimId));
		assert(childDimId != rootDimId);

		const double subtreeExpEntropy = entropy(subtreeJointPartitions[childDimId], true);
		const dim_t subtreeDims = subtreeDimIds[childDimId].size();
		const double subtreeCost = subspaceCost(subtreeExpEntropy, subtreeDims, trueNumVectors);

		const double complementExpEntropy = entropy(complementJointPartitions[childDimId], true);
		const dim_t complementDims = complementDimIds[childDimId].size();
		const double complementCost = subspaceCost(complementExpEntropy, complementDims, trueNumVectors);

		const double changeInTotalCost = subtreeCost + complementCost - subspaceIdToCost[subspaceId];

		// The edge that we are splitting on is the edge from the child to the parent
		// It doesn't matter if the first dimension ID of the edge is smaller for now (TODO)
		const std::pair<dim_t, dim_t> edge{ childDimId, parentDimId };

		splitQueue.emplace_back(
			SplitOperation{
				changeInTotalCost,
				edge,
				subtreeDimIds[childDimId],
				complementDimIds[childDimId],
				subtreeCost,
				complementCost
			}
		);
	}
}

GreedySplitter::GreedySplitter(SubspaceDecomposition& decomp, const Graph& tree, const std::vector<Partition>& partitions, const Parameters& params)
	: pDecomp(&decomp), pTree(&tree), pPartitions(&partitions), trueNumVectors(params.trueNumVectors)
{
	for (const dim_t& subspaceId : decomp.allSubspaceIds())
	{
		// Add all splits of the subspace to the queue and update the subspaceIdToCost map
		enqueueSubspace(subspaceId);
	}

	// Set loss of 0-th iteration to the total cost of the decomposition
	losses.push_back(std::accumulate(subspaceIdToCost.begin(), subspaceIdToCost.end(),
									 0.0, [](double sum, const auto& pair) { return sum + pair.second; }));
}

bool GreedySplitter::canSplit() const
{
	if (splitQueue.empty())
	{
		return false;
	}

	const SplitOperation& bestSplit = *std::min_element(splitQueue.begin(), splitQueue.end());
	return bestSplit.changeInTotalCost < 0.0;
}

void GreedySplitter::doSplit()
{
	// For sanity check:
	const dim_t splitQueueSizeBefore = splitQueue.size();

	// Get the best split operation from the queue
	assert(canSplit());
	std::vector<SplitOperation>::iterator bestSplitIt = std::min_element(splitQueue.begin(), splitQueue.end());
	SplitOperation bestSplit = std::move(*bestSplitIt);
	splitQueue.erase(bestSplitIt);

	// Lookup the subspace
	const dim_t subspaceId = pDecomp->getSubspaceId(bestSplit.edge.first);
	assert(subspaceId == pDecomp->getSubspaceId(bestSplit.edge.second));

	// Clean up the old subspace from the cost map
	subspaceIdToCost.erase(subspaceId);

	// Remove all of the now invalidated splits of the subspace from queue
	// TODO: Some partitions in the queue can be reused to save time, but for now we just remove them all
	auto it = splitQueue.begin();
	while (it != splitQueue.end())
	{
		if (pDecomp->inSameSubspace(it->edge.first, subspaceId))
		{
			assert(it->edge.second == subspaceId || pDecomp->inSameSubspace(it->edge.second, subspaceId));
			it = splitQueue.erase(it);
		}
		else
		{
			++it;
		}
	}

	// Do the split
	assert(pDecomp->inSameSubspace(bestSplit.edge.first, bestSplit.edge.second));
	pDecomp->split(bestSplit.dimIds1, bestSplit.dimIds2);
	assert(!pDecomp->inSameSubspace(bestSplit.edge.first, bestSplit.edge.second));

	// For each of the two new subspaces, add all splits to the queue and update the cost map
	enqueueSubspace(pDecomp->getSubspaceId(bestSplit.edge.first));
	enqueueSubspace(pDecomp->getSubspaceId(bestSplit.edge.second));

	// Record the loss of the current iteration
	losses.push_back(losses.back() + bestSplit.changeInTotalCost);

	// Sanity check
	assert(splitQueue.size() == splitQueueSizeBefore - 1);
}

std::vector<double> GreedySplitter::getLosses() const
{
	return losses;
}

} // namespace npq