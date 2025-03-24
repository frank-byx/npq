/**
 * @brief Implementation of the O(kn) time and O(n) space dynamic programming algorithm
 * for optimal 1-dimensional k-means clustering described in section 3 of:
 * Grønlund, A., Larsen, K. G., Mathiasen, A., Nielsen, J. S., Schneider, S., & Song, M. (2018).
 * Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D. arXiv. https://arxiv.org/abs/1701.07204
 */

#pragma once

#include <array>
#include <vector>

#include "dataset.h"
#include "partition.h"


namespace npq
{

	/**
	* @brief Data structure that computes the cost of a cluster in O(1) time.
	*/
	class CostCalculator {
	private:
		/**
		* @brief The prefix sum array of the input data in sorted order.
		*/
		std::vector<double> cumsum;

		/**
		* @brief The prefix sum array of the squares of the input data in sorted order.
		*/
		std::vector<double> cumsum2;

	public:
		/**
		* Default constructor that doesn't properly initialize the object.
		*/
		CostCalculator() = default;

		/**
		* @brief Move constructor that takes the prefix sum arrays as input.
		*/
		CostCalculator(std::vector<double>&& cumsum, std::vector<double>&& cumsum2);

		/**
		* @brief Computes the cost of cluster/interval [i, j] (of the data in sorted order).
		*/
		float operator()(id_t i, id_t j) const;
	};


	/**
	 * @brief The state of the dynamic programming algorithm for 1D k-means clustering.
	 *
	 * Instead of storing the entire DP matrix D, we only store the last two rows of D.
	 * D[k][m] is the cost of optimally clustering x_0,...,x_{m-1} into k clusters.
	 */
	struct DPState
	{
		/**
		 * @brief Pointer to the input 1D data to be clustered.
		 */
		const std::vector<float>* pData;

		/**
		* @brief Stores the sorted order of the data.
		*
		* The i-th value in this list is the index of the i-th smallest element in the data (0-th is smallest).
		*/
		std::vector<id_t> sortedOrder;

		/**
		* @brief The cost function of a cluster.
		*/
		CostCalculator cc;

		/**
		 * @brief The last two filled rows of the DP matrix D at any given iteration.
		 *
		 * The row at index 0 is the second-last row, and the row at index 1 is the last row.
		 */
		std::array<std::vector<float>, 2> lastTwoRows;
	};


	/**
	 * @brief Initializes the DP state with the input data.
	 *
	 * Allocates memory, computes the sorted order of the data, and computes prefix sum arrays for the cost function.
	 */
	void initializeDPState(DPState& state, const std::vector<float>& data);

	/**
	 * @brief Performs a single iteration of the dynamic programming algorithm for 1D k-means clustering.
	 *
	 * This function computes the next row of the DP matrix D from the last row, and keeps the last two rows.
	 *
	 * @param state The current DP state object.
	 */
	void doDPIteration(DPState& state);

	/**
	 * @brief Computes the optimal clustering/paritition corresponding to the second-last row of the DP table.
	 *
	 * After computing k+1 rows of the DP table, this function uses the regularized k-means algorithm described in
	 * the paper to compute the optimal k-clustering from OPT_k and OPT_{k+1} instead of using backtracking.
	 *
	 * @param state The current DP state object.
	 */
	Partition computePartitionFromDPState(DPState& state);

	/**
	 * @brief Computes the entropy of the corresponding optimal clustering/paritition.
	 *
	 * This is equivalent to calling entropy(computePartitionFromDPState(state)), but is more efficient.
	 *
	 * @param state The current DP state object.
	 */
	double computeEntropyFromDPState(DPState& state);

} // namespace npq