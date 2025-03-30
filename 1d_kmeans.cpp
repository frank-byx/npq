#include <algorithm>
#include <limits>
#include <numeric>

#include "1d_kmeans.h"
#include "smawk.h"
#include "regularized_1d_kmeans.h"


namespace npq
{

	CostCalculator::CostCalculator(std::vector<double>&& cumsum, std::vector<double>&& cumsum2)
		: cumsum{ std::move(cumsum) }, cumsum2{ std::move(cumsum2) }
	{}

	float CostCalculator::operator()(id_t i, id_t j) const
	{
		if (j < i)
		{
			return 0.0f;
		}

		auto mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1);

		auto result = cumsum2[j + 1] - cumsum2[i];
		result += (j - i + 1) * (mu * mu);
		result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);

		return float(result);
	}


	void initializeDPState(DPState& state, const std::vector<float>& data)
	{
		// Store a pointer to the data
		state.pData = &data;
		const id_t n = data.size();

		// Compute the sorted order of the data
		state.sortedOrder.resize(n);
		std::iota(state.sortedOrder.begin(), state.sortedOrder.end(), 0);
		std::sort(state.sortedOrder.begin(), state.sortedOrder.end(), [&](id_t i, id_t j) { return data[i] < data[j]; });

		// Compute the prefix sum arrays of the sorted data and construct the cost function
		std::vector<double> cumsum = { 0.0 };
		std::vector<double> cumsum2 = { 0.0 };
		cumsum.reserve(n + 1);
		cumsum2.reserve(n + 1);
		for (id_t i = 0; i < n; ++i)
		{
			const float x = data[state.sortedOrder[i]];
			cumsum.push_back(x + cumsum[i]);
			cumsum2.push_back(x * x + cumsum2[i]);
		}
		state.cc = CostCalculator(std::move(cumsum), std::move(cumsum2));

		// Allocate memory for the last two rows of the DP matrix D, and fill out the first row (in row 1)
		state.lastTwoRows[0].resize(n);
		state.lastTwoRows[1].resize(n);
		for (id_t m = 0; m < n; ++m)
		{
			state.lastTwoRows[1][m] = state.cc(0, m);
		}
	}

	void doDPIteration(DPState& state)
	{
		/******************************************************************************************
		N: number of points
		K: number of clusters
		CC(i, j): the cost of grouping xi,...,xj into one cluster
		D[k][m]:  the cost of optimally clustering x1,...,xm into k clusters

		Assume the data is already sorted in ascending order. The algorithm is:

		For k from 0 to K-1:
			For m from 0 to N-1:
				Compute D[k][m] = min_i D[k - 1][i - 1] + CC(i, m)

		Return D[K-1][N-1]

		We use the SMAWK algorithm to compute the min for all m from 0 to N-1 in O(N) time.
		To do so, for each k we define an implicit totally monotone matrix C:

			C[m][i] = D[k - 1][i - 1] + CC(i, m)

		And then we apply the SMAWK algorithm to find the row minima of C.

		The overall time complexity of the algorithm is thus O(KN),
		and the time complexity of a single call to this function is O(N).
		******************************************************************************************/

		auto C = [&state](id_t m, id_t i) {
			if (i == 0)
			{
				return state.cc(i, m);
			}
			id_t col = std::min(m, i - 1);
			return state.lastTwoRows[1][col] + state.cc(i, m);
			};

		const id_t n = state.pData->size();
		std::vector<id_t> argmins(n);
		smawk(n, C, argmins.data());
		for (id_t m = 0; m < n; m++) {
			id_t idx = argmins[m];
			state.lastTwoRows[0][m] = C(m, idx);  // Overwrite the second-last row
		}

		// Swap the last two rows since we overwrote the second-last row
		state.lastTwoRows[0].swap(state.lastTwoRows[1]);
	}

	Partition computePartitionFromDPState(DPState& state)
	{
		// Do regularized 1D k-means
		const id_t n = state.pData->size();
		double lambda = state.lastTwoRows[0][n - 1] - state.lastTwoRows[1][n - 1];
		lambda += std::numeric_limits<scalar_t>::epsilon();  // Add epsilon to avoid issue with duplicate values
		std::vector<id_t> path = computeRegularized1DKMeans(lambda, state.cc, n);

		// Convert the output path of leftmost elements into a Partition object
		const id_t k = path.size();
		Partition partition(n, k);
		for (id_t j = 0; j < k; ++j)
		{
			const id_t left = path[j];
			const id_t right = j + 1 < k ? path[j + 1] - 1 : n - 1;
			for (id_t i = left; i <= right; ++i)
			{
				const id_t vecId = state.sortedOrder[i];
				const id_t block_id = j;
				partition.vecIdToBlockId[vecId] = block_id;
				partition.blockIdToVecIds[block_id].push_back(vecId);
			}
		}

		return partition;
	}

	double computeEntropyFromDPState(DPState& state)
	{
		// Do regularized 1D k-means
		const id_t n = state.pData->size();
		double lambda = state.lastTwoRows[0][n - 1] - state.lastTwoRows[1][n - 1];
		std::vector<id_t> path = computeRegularized1DKMeans(lambda, state.cc, n);

		// Compute the entropy of the partition directly from the output path
		double acc = 0.0;
		const id_t k = path.size();
		for (id_t j = 0; j < k; ++j)
		{
			const id_t left = path[j];
			const id_t right = j + 1 < k ? path[j + 1] - 1 : n - 1;
			const id_t length = right - left + 1;

			const double p = static_cast<double>(length) / n;
			acc -= p * std::log2(p);
		}

		return acc;
	}

} // namespace npq