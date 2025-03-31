/**
 * @brief The first step of the algorithm. See algorithm_steps.h for more information.
 */

#include <cassert>
#include <cmath>
#include <queue>

#include "algorithm_steps.h"
#include "1d_kmeans.h"


namespace npq
{

	std::vector<Partition> compute1dPartitions(const Dataset& dataset, const Parameters& params)
	{
		// Subspace storage cost function simplified for the case of a single dimension
		auto subspaceStorageCost1d = [&params](id_t numClusters) {
			const double scalarSize = sizeof(scalar_t) * 8;
			return params.trueNumVectors * log2(numClusters) + numClusters * scalarSize;
		};

		const dim_t d = dataset.dimensions.size();
		std::vector<DPState> dpStates;
		dpStates.resize(d);

		// TODO: Update to use normalized target distortion
		const id_t n = dataset.dimensions[0].size();
		const double targetTotalSquaredError = params.targetDistortion * (1.0 - params.targetDistortionMargin) * n;
		double totalSquaredError = 0.0;

		// Create a priority queue of dimensions by highest efficiency, where efficiency of a dimension
		// is defined as the decrease in total squared error divided by the increase in storage cost
		// from adding a cluster to that dimension
		std::priority_queue<
			std::pair<double, dim_t>,
			std::vector<std::pair<double, dim_t>>
		> pq;

		// Initialization
		for (dim_t i = 0; i < d; ++i)
		{
			// Initialize the DP state for dimension i and compute the initial total squared error
			initializeDPState(dpStates[i], dataset.dimensions[i]);
			totalSquaredError += dpStates[i].lastTwoRows[0].back();

			// Compute the efficiency of adding a cluster to dimension i and push to the priority queue
			const double decreaseInTotalSquaredError = dpStates[i].lastTwoRows[0].back() - dpStates[i].lastTwoRows[1].back();
			assert(decreaseInTotalSquaredError >= 0.0);

			const double k = dpStates[i].numClusters;
			assert(k == 1);
			const double increaseInStorageCost = subspaceStorageCost1d(k + 1) - subspaceStorageCost1d(k);
			assert(increaseInStorageCost > 0.0);

			const double efficiency = decreaseInTotalSquaredError / increaseInStorageCost;
			pq.push({ efficiency, i });
		}

		// While the total squared error is above the target, greedily add a cluster to the most efficient dimension
		while (totalSquaredError > targetTotalSquaredError)
		{
			// Pop the most efficient dimension from the priority queue
			const dim_t i = pq.top().second;
			pq.pop();

			// Add a cluster to dimension i and update the total squared error
			const double oldSquaredError = dpStates[i].lastTwoRows[0].back();
			doDPIteration(dpStates[i]);
			const double newSquaredError = dpStates[i].lastTwoRows[0].back();
			totalSquaredError -= oldSquaredError - newSquaredError;

			// Compute the efficiency of adding another cluster to dimension i and push back to the priority queue
			const double decreaseInTotalSquaredError = dpStates[i].lastTwoRows[0].back() - dpStates[i].lastTwoRows[1].back();
			assert(decreaseInTotalSquaredError >= 0.0);

			const double k = dpStates[i].numClusters;
			const double increaseInStorageCost = subspaceStorageCost1d(k + 1) - subspaceStorageCost1d(k);
			assert(increaseInStorageCost > 0.0);

			const double efficiency = decreaseInTotalSquaredError / increaseInStorageCost;
			pq.push({ efficiency, i });
		}

		// Create the final partitions from the DP states
		std::vector<Partition> partitions;
		partitions.reserve(d);
		for (dim_t i = 0; i < d; ++i)
		{
			partitions.emplace_back(computePartitionFromDPState(dpStates[i]));
		}

		return partitions;
	}

} // namespace npq


// TODO: Add optimizations to avoid recomputing the path the last time, and entropies in the next step
