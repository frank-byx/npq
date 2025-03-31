/**
 * @brief The third step of the algorithm. See algorithm_steps.h for more information.
 */

#include <iostream>
#include <string>

#include "algorithm_steps.h"
#include "greedy.h"


namespace npq
{

SubspaceDecomposition computeSubspaceDecomposition(
	const std::vector<Partition>& partitions,
	const Graph& tree,
	const Parameters& params
)
{
	const dim_t numDims = partitions.size();

	// Intialize a decomposition of singletons to merge, and a decomposition with a single subspace to split
	SubspaceDecomposition decompToMerge(numDims, true);
	SubspaceDecomposition decompToSplit(numDims, false);
	double decompToMergeCost = -1.0;
	double decompToSplitCost = -1.0;
	std::string decompToMergeLabel = "Decomposition A";
	std::string decompToSplitLabel = "Decomposition B";

	// Run greedy merger and splitter on their respective decompositions until they can't merge/split anymore,
	// and then swap their decompositions and repeat until both still have no moves after swapping.
	// When we stop, the decompositions may or may not have converged to the same decomposition.
	int swapCount = 0;
	bool didMerge = true;
	bool didSplit = true;
	while (decompToMerge != decompToSplit && (didMerge || didSplit))
	{
		std::vector<double> mergerLosses;
		std::vector<double> splitterLosses;

		// Run greedy merger and splitter (in separate scopes to minimize peak memory usage)
		{
			GreedyMerger merger(decompToMerge, tree, partitions, params);
			didMerge = false;
			while (merger.canMerge())
			{
				merger.doMerge();
				didMerge = true;
			}

			mergerLosses = merger.getLosses();
			decompToMergeCost = mergerLosses.back();
		}
		{
			GreedySplitter splitter(decompToSplit, tree, partitions, params);
			didSplit = false;
			while (splitter.canSplit())
			{
				splitter.doSplit();
				didSplit = true;
			}

			splitterLosses = splitter.getLosses();
			decompToSplitCost = splitterLosses.back();
		}

		// Print losses
		std::cout << decompToMergeLabel << " cost at start of round " << swapCount << ": " << mergerLosses[0] << std::endl;
		for (size_t i = 1; i < mergerLosses.size(); ++i)
		{
			std::cout << "Cost after " << i << " merge(s): " << mergerLosses[i] << std::endl;
		}
		std::cout << std::endl;
		
		std::cout << decompToSplitLabel << " cost at start of round " << swapCount << ": " << splitterLosses[0] << std::endl;
		for (size_t i = 1; i < splitterLosses.size(); ++i)
		{
			std::cout << "Cost after " << i << " split(s): " << splitterLosses[i] << std::endl;
		}
		std::cout << std::endl;

		// Swap merging and splitting decompositions
		std::swap(decompToMerge, decompToSplit);
		std::swap(decompToMergeCost, decompToSplitCost);
		std::swap(decompToMergeLabel, decompToSplitLabel);
		++swapCount;
	}

	// Return the decomposition with the lower cost
	if (decompToMerge == decompToSplit)
	{
		std::cout << "Decompositions A and B converged to the same decomposition." << std::endl;
		std::cout << "The cost of the final decomposition is: " << decompToMergeCost << std::endl;

		return decompToMerge;
	}
	else
	{
		std::cout << "Decompositions A and B converged to different decompositions." << std::endl;
		std::cout << "The cost of " << decompToMergeLabel << " is: " << decompToMergeCost << std::endl;
		std::cout << "The cost of " << decompToSplitLabel << " is: " << decompToSplitCost << std::endl;

		if (decompToMergeCost < decompToSplitCost)
		{
			std::cout << "Returning " << decompToMergeLabel << "." << std::endl;
			return decompToMerge;
		}
		else
		{
			std::cout << "Returning " << decompToSplitLabel << "." << std::endl;
			return decompToSplit;
		}
	}
}

} // namespace npq