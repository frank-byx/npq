// NOTE: The only local dependencies shared with the main algorithm are dataset.h and partition.h.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "naive_npq.h"
#include "partition.h"


namespace npq
{

// The code in the following section is adapted from code from the FAISS repository licensed under the MIT license.
// ====================================================================================================================

using LookUpFunc = std::function<float(id_t, id_t)>;

void reduce(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const LookUpFunc& lookup,
    std::vector<id_t>& output_cols) {
    for (id_t col : input_cols) {
        while (!output_cols.empty()) {
            id_t row = rows[output_cols.size() - 1];
            float a = lookup(row, col);
            float b = lookup(row, output_cols.back());
            if (a >= b) { // defeated
                break;
            }
            output_cols.pop_back();
        }
        if (output_cols.size() < rows.size()) {
            output_cols.push_back(col);
        }
    }
}

void interpolate(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& cols,
    const LookUpFunc& lookup,
    id_t* argmins) {
    std::unordered_map<id_t, id_t> id_to_col;
    for (id_t idx = 0; idx < cols.size(); ++idx) {
        id_to_col[cols[idx]] = idx;
    }

    id_t start = 0;
    for (id_t r = 0; r < rows.size(); r += 2) {
        id_t row = rows[r];
        id_t end = cols.size() - 1;
        if (r < rows.size() - 1) {
            id_t idx = argmins[rows[r + 1]];
            end = id_to_col[idx];
        }
        id_t argmin = cols[start];
        float min = lookup(row, argmin);
        for (id_t c = start + 1; c <= end; c++) {
            float value = lookup(row, cols[c]);
            if (value < min) {
                argmin = cols[c];
                min = value;
            }
        }
        argmins[row] = argmin;
        start = end;
    }
}

/** SMAWK algo. Find the row minima of a monotone matrix.
 *
 * References:
 *   1. http://web.cs.unlv.edu/larmore/Courses/CSC477/monge.pdf
 *   2. https://gist.github.com/dstein64/8e94a6a25efc1335657e910ff525f405
 *   3. https://github.com/dstein64/kmeans1d
 */
void smawk_impl(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const LookUpFunc& lookup,
    id_t* argmins) {
    if (rows.size() == 0) {
        return;
    }

    /**********************************
     * REDUCE
     **********************************/
    auto ptr = &input_cols;
    std::vector<id_t> survived_cols; // survived columns
    if (rows.size() < input_cols.size()) {
        reduce(rows, input_cols, lookup, survived_cols);
        ptr = &survived_cols;
    }
    auto& cols = *ptr; // avoid memory copy

    /**********************************
     * INTERPOLATE
     **********************************/

     // call recursively on odd-indexed rows
    std::vector<id_t> odd_rows;
    for (id_t i = 1; i < rows.size(); i += 2) {
        odd_rows.push_back(rows[i]);
    }
    smawk_impl(odd_rows, cols, lookup, argmins);

    // interpolate the even-indexed rows
    interpolate(rows, cols, lookup, argmins);
}

void smawk(
    const id_t nrows,
    const id_t ncols,
    const LookUpFunc& lookup,
    id_t* argmins) {
    std::vector<id_t> rows(nrows);
    std::vector<id_t> cols(ncols);
    std::iota(std::begin(rows), std::end(rows), 0);
    std::iota(std::begin(cols), std::end(cols), 0);

    smawk_impl(rows, cols, lookup, argmins);
}

void smawk(
    const id_t nrows,
    const id_t ncols,
    const float* x,
    id_t* argmins) {
    auto lookup = [&x, &ncols](id_t i, id_t j) { return x[i * ncols + j]; };
    smawk(nrows, ncols, lookup, argmins);
}

namespace {

    class CostCalculator {
        // The reuslt would be inaccurate if we use float
        std::vector<double> cumsum;
        std::vector<double> cumsum2;

    public:
        CostCalculator(const std::vector<float>& vec, id_t n) {
            cumsum.push_back(0.0);
            cumsum2.push_back(0.0);
            for (id_t i = 0; i < n; ++i) {
                float x = vec[i];
                cumsum.push_back(x + cumsum[i]);
                cumsum2.push_back(x * x + cumsum2[i]);
            }
        }

        float operator()(id_t i, id_t j) {
            if (j < i) {
                return 0.0f;
            }
            auto mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1);
            auto result = cumsum2[j + 1] - cumsum2[i];
            result += (j - i + 1) * (mu * mu);
            result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);
            return float(result);
        }
    };

    template <class T>
    class Matrix {
        std::vector<T> data;
        id_t nrows;
        id_t ncols;

    public:
        Matrix(id_t nrows, id_t ncols) {
            this->nrows = nrows;
            this->ncols = ncols;
            data.resize(nrows * ncols);
        }

        inline T& at(id_t i, id_t j) {
            return data[i * ncols + j];
        }

        void addRow()
        {
            ++(this->nrows);
            data.resize(nrows * ncols);
        }
    };

} // anonymous namespace

// This function is modified to take the target cost as input instead of the number of clusters,
// and instead of returning the centroids it returns the clusters as a list of sets of indices
std::vector<std::vector<id_t>> kmeans1d(const float* x, size_t n, double targetCost) {
    std::vector<float> arr(x, x + n);
	std::vector<id_t> indices(n);
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&arr](id_t i, id_t j) { return arr[i] < arr[j]; });
    std::sort(arr.begin(), arr.end());

    /***************************************************
    dynamic programming algorithm

    Reference: https://arxiv.org/abs/1701.07204
    -------------------------------

    Assume x is already sorted in ascending order.

    N: number of points
    K: number of clusters

    CC(i, j): the cost of grouping xi,...,xj into one cluster
    D[k][m]:  the cost of optimally clustering x1,...,xm into k clusters
    T[k][m]:  the start index of the k-th cluster

    The DP process is as follow:
        D[k][m] = min_i D[k - 1][i - 1] + CC(i, m)
        T[k][m] = argmin_i D[k - 1][i - 1] + CC(i, m)

    This could be solved in O(KN^2) time and O(KN) space.

    To further reduce the time complexity, we use SMAWK algo to
    solve the argmin problem as follow:

    For each k:
        C[m][i] = D[k - 1][i - 1] + CC(i, m)

        Here C is a n x n totally monotone matrix.
        We could find the row minima by SMAWK in O(N) time.

    Now the time complexity is reduced from O(kN^2) to O(KN).
    ****************************************************/

    CostCalculator CC(arr, n);
    Matrix<float> D(1, n);
    Matrix<id_t> T(1, n);

    for (id_t m = 0; m < n; m++) {
        D.at(0, m) = CC(0, m);
        T.at(0, m) = 0;
    }

    id_t k = 0;
    while (D.at(k, n - 1) > targetCost) {
        D.addRow();
        T.addRow();
        ++k;

        // we define C here
        auto C = [&D, &CC, &k](id_t m, id_t i) {
            if (i == 0) {
                return CC(i, m);
            }
            id_t col = std::min(m, i - 1);
            return D.at(k - 1, col) + CC(i, m);
        };

        std::vector<id_t> argmins(n); // argmin of each row
        smawk(n, n, C, argmins.data());
        for (id_t m = 0; m < argmins.size(); m++) {
            id_t idx = argmins[m];
            D.at(k, m) = C(m, idx);
            T.at(k, m) = idx;
        }
    }

    /***************************************************
    compute clusters by backtracking

           T[K - 1][T[K][N] - 1]        T[K][N]        N
    --------------|------------------------|-----------|
                  |     cluster K - 1      | cluster K |

    ****************************************************/

    std::vector<std::vector<id_t>> clusters;
    clusters.reserve(k + 1);
    id_t end = n;
    for (; k >= 0; k--) {
		std::vector<id_t> cluster;
        const id_t start = T.at(k, end - 1);
		for (id_t i = start; i < end; ++i) {
			cluster.push_back(indices[i]);
		}
        assert(cluster.size() > 0);
		clusters.push_back(std::move(cluster));
        end = start;
    }

	return clusters;
}

// ====================================================================================================================
// End of code adapted from the FAISS repository

// This function is copied from greedy.cpp
double subspaceCost(double expEntropy, dim_t subspaceDims, id_t numVectors)
{
    const double scalarSize = sizeof(scalar_t) * 8;
    return numVectors * log2(expEntropy) + expEntropy * subspaceDims * scalarSize;
}

// Merge operation (changeInTotalCost, (subspaceId1, subspaceId2)), where subspaceId1 < subspaceId2
using MergeOp = std::pair<double, std::pair<dim_t, dim_t>>;

std::vector<std::vector<dim_t>> naiveNPQ(const Dataset& dataset, double targetDistortion)
{
    std::cout << "Running naiveNPQ." << std::endl;

	const dim_t numDims = dataset.dimensions.size();
    const id_t numVectors = dataset.dimensions[0].size();
    std::cout << "numDims: " << numDims << ", numVectors: " << numVectors << std::endl;

    std::cout << "Computing 1D partitions." << std::endl;

	const double target1dError = targetDistortion / numDims;  // Balance the MSE of each 1D partition
	std::vector<Partition> partitions;
	partitions.reserve(numDims);
    for (dim_t i = 0; i < numDims; ++i)
    {
		const float* dimData = dataset.dimensions[i].data();
        std::vector<std::vector<id_t>> blockIdToVecIds = kmeans1d(dimData, numVectors, target1dError);

		std::vector<id_t> vecIdToBlockId;
		vecIdToBlockId.resize(numVectors, -1);
        for (id_t blockId = 0; blockId < blockIdToVecIds.size(); ++blockId)
        {
            for (const id_t& vecId : blockIdToVecIds[blockId])
            {
				assert(vecIdToBlockId[vecId] == -1);
                vecIdToBlockId[vecId] = blockId;
            }
        }
		for (id_t vecId = 0; vecId < numVectors; ++vecId)
		{
			assert(vecIdToBlockId[vecId] != -1);
		}

		partitions.emplace_back(std::move(vecIdToBlockId), std::move(blockIdToVecIds));
    }

	std::cout << "Computing subspace decomposition by naive greedy merging." << std::endl;

	// Initial subspace decomposition of singleton dimensions
    std::vector<dim_t> dimIdToSubspaceId;
	dimIdToSubspaceId.resize(numDims);
	std::iota(dimIdToSubspaceId.begin(), dimIdToSubspaceId.end(), 0);

    // Set the maps for the initial subspace decomposition
	std::map<dim_t, Partition> subspaceIdToJointPartition;
	std::map<dim_t, double> subspaceIdToCost;
	for (dim_t subspaceId = 0; subspaceId < numDims; ++subspaceId)
	{
		subspaceIdToJointPartition[subspaceId] = std::move(partitions[subspaceId]);
		const double subspaceExpEntropy = entropy(subspaceIdToJointPartition[subspaceId], true);
		subspaceIdToCost[subspaceId] = subspaceCost(subspaceExpEntropy, 1, numVectors);
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
			const double newCost = subspaceCost(newExpEntropy, 2, numVectors);
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
		subspaceIdToCost[subspaceId1] = subspaceCost(newExpEntropy, newSubspaceDims, numVectors);

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
            const double opCost = subspaceCost(opExpEntropy, opNumDims, numVectors);
            op.first = opCost - subspaceIdToCost[op.second.first] - subspaceIdToCost[op.second.second];
        }

		// Record the loss of the current iteration
        loss = loss + bestMerge.first;
		++mergeCount;
		std::cout << "Cost after " << mergeCount << " merge(s): " << loss << std::endl;
    }

	// Return the subspace decomposition
    std::vector<dim_t> subspaceIds = dimIdToSubspaceId;
    std::sort(subspaceIds.begin(), subspaceIds.end());
    subspaceIds.erase(std::unique(subspaceIds.begin(), subspaceIds.end()), subspaceIds.end());

	std::vector<std::vector<dim_t>> subspaces;
	subspaces.reserve(subspaceIds.size());
	for (const dim_t& subspaceId : subspaceIds)
	{
		std::vector<dim_t> dimIds;
		for (dim_t dimId = 0; dimId < numDims; ++dimId)
		{
			if (dimIdToSubspaceId[dimId] == subspaceId)
			{
				dimIds.push_back(dimId);
			}
		}
		subspaces.push_back(std::move(dimIds));
	}

    return subspaces;
}

} // namespace npq
