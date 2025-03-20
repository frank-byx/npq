/**
 * @brief The first step of the algorithm. See algorithm_steps.h for more information.
 */

#include <queue>

#include "algorithm_steps.h"
#include "1d_kmeans.h"


namespace npq
{

	std::vector<Partition> Compute1dPartitions(const Dataset& dataset, const Parameters& params)
	{
		const dim_t d = dataset.dimensions.size();
		std::vector<DPState> dpStates(d);

		// TODO: Update to use normalized target distortion
		const double maxCost = params.targetDistortion - params.targetDistortionMargin;
		double totalCost = 0.0;

		// Create a priority queue of dimensions by lowest entropy of
		// the optimal clustering corresponding to its current DP table)
		std::priority_queue<
			std::pair<double, dim_t>,
			std::vector<std::pair<double, dim_t>>,
			std::greater<std::pair<double, dim_t>>
		> pq;

		// For each dimension, initialize the DP state, do one iteration, and push to pq
		for (dim_t i = 0; i < d; ++i)
		{
			InitializeDPState(dpStates[i], dataset.dimensions[i]);
			DoDPIteration(dpStates[i]);

			const double cost = dpStates[i].lastTwoRows[0].back();
			totalCost += cost;

			const double entropy = ComputeEntropyFromDPState(dpStates[i]);
			pq.push({ entropy, i });
		}

		// While the total cost is above the maximum allowed cost,
		// pop the dimension with the lowest entropy and do an iteration of DP
		while (totalCost > maxCost)
		{
			const dim_t i = pq.top().second;
			pq.pop();

			const double oldCost = dpStates[i].lastTwoRows[0].back();
			DoDPIteration(dpStates[i]);
			const double newCost = dpStates[i].lastTwoRows[0].back();
			totalCost -= oldCost - newCost;

			const double entropy = ComputeEntropyFromDPState(dpStates[i]);
			pq.push({ entropy, i });
		}

		// Create the final partitions from the DP states
		std::vector<Partition> partitions;
		partitions.reserve(d);
		for (dim_t i = 0; i < d; ++i)
		{
			partitions.emplace_back(ComputePartitionFromDPState(dpStates[i]));
		}

		return partitions;
	}

} // namespace npq


// TODO: Add optimizations to avoid recomputing the path the last time, and entropies in the next step
