#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>

#include "1d_kmeans.h"
#include "smawk.h"
#include "regularized_1d_kmeans.h"


// TODO: The following is only included as a temporary replacement for the regularized algorithm
// The code in the following section is adapted from code from the FAISS repository licensed under the MIT license.
// ====================================================================================================================

using namespace npq;

using _LookUpFunc = std::function<float(id_t, id_t)>;

void _reduce(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const _LookUpFunc& lookup,
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

void _interpolate(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& cols,
    const _LookUpFunc& lookup,
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
void _smawk_impl(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const _LookUpFunc& lookup,
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
        _reduce(rows, input_cols, lookup, survived_cols);
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
    _smawk_impl(odd_rows, cols, lookup, argmins);

    // interpolate the even-indexed rows
    _interpolate(rows, cols, lookup, argmins);
}

void _smawk(
    const id_t nrows,
    const id_t ncols,
    const _LookUpFunc& lookup,
    id_t* argmins) {
    std::vector<id_t> rows(nrows);
    std::vector<id_t> cols(ncols);
    std::iota(std::begin(rows), std::end(rows), 0);
    std::iota(std::begin(cols), std::end(cols), 0);

    _smawk_impl(rows, cols, lookup, argmins);
}

void _smawk(
    const id_t nrows,
    const id_t ncols,
    const float* x,
    id_t* argmins) {
    auto lookup = [&x, &ncols](id_t i, id_t j) { return x[i * ncols + j]; };
    _smawk(nrows, ncols, lookup, argmins);
}

namespace {

    class _CostCalculator {
        // The reuslt would be inaccurate if we use float
        std::vector<double> cumsum;
        std::vector<double> cumsum2;

    public:
        _CostCalculator(const std::vector<float>& vec, id_t n) {
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
    class _Matrix {
        std::vector<T> data;
        id_t nrows;
        id_t ncols;

    public:
        _Matrix(id_t nrows, id_t ncols) {
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
std::vector<std::vector<id_t>> kmeans1dTemp(const float* x, size_t n, id_t targetK) {
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

    _CostCalculator CC(arr, n);
    _Matrix<float> D(1, n);
    _Matrix<id_t> T(1, n);

    for (id_t m = 0; m < n; m++) {
        D.at(0, m) = CC(0, m);
        T.at(0, m) = 0;
    }

    id_t k = 0;
    while (k + 1 < targetK) {
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
        _smawk(n, n, C, argmins.data());
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

		// Initialize the number of clusters to 0, which is currently invalid,
		// so that after the first iteration it will be 1
		state.numClusters = 0;

		// Do the first iteration of the DP algorithm, which will compute the second row of the DP matrix D
		// and move the first row to lastTwoRows[0], making the DP state valid and initialized to a single cluster
		doDPIteration(state);
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

		// Increment the number of clusters
		++state.numClusters;
	}

	Partition computePartitionFromDPState(DPState& state)
	{
        // TODO: Fix the regularized algorithm and uncomment the following:
		// Do regularized 1D k-means
		//const id_t n = state.pData->size();
		//double lambda = state.lastTwoRows[0][n - 1] - state.lastTwoRows[1][n - 1];
		//lambda += std::numeric_limits<scalar_t>::epsilon();  // Add epsilon to avoid issue with duplicate values
		//std::vector<id_t> path = computeRegularized1DKMeans(lambda, state.cc, n);

		// Convert the output path of leftmost elements into a Partition object
		//const id_t k = path.size();
		//Partition partition(n, k);
		//for (id_t j = 0; j < k; ++j)
		//{
		//	const id_t left = path[j];
		//	const id_t right = j + 1 < k ? path[j + 1] - 1 : n - 1;
		//	std::cout << left << " " << right << std::endl;
		//	for (id_t i = left; i <= right; ++i)
		//	{
		//		const id_t vecId = state.sortedOrder[i];
		//		const id_t block_id = j;
		//		partition.vecIdToBlockId[vecId] = block_id;
		//		partition.blockIdToVecIds[block_id].push_back(vecId);
		//	}
		//}

		// TODO: The the number of partition blocks and the number of clusters should always be equal, but there is a bug
		//       in the regularized algorithm. This causes the algorithm to overshoot the target quantization error.
		//assert(partition.blockIdToVecIds.size() - state.numClusters <= 1);
		
		// TODO: For now we have swapped out the regularized algorithm
        const id_t n = state.pData->size();
		const float* dimData = state.pData->data();
		std::vector<std::vector<id_t>> blockIdToVecIds = kmeans1dTemp(dimData, n, state.numClusters);

		std::vector<id_t> vecIdToBlockId;
		vecIdToBlockId.resize(n, -1);
		for (id_t blockId = 0; blockId < blockIdToVecIds.size(); ++blockId)
		{
			for (const id_t& vecId : blockIdToVecIds[blockId])
			{
				assert(vecIdToBlockId[vecId] == -1);
				vecIdToBlockId[vecId] = blockId;
			}
		}
		for (id_t vecId = 0; vecId < n; ++vecId)
		{
			assert(vecIdToBlockId[vecId] != -1);
		}

		Partition partition { std::move(vecIdToBlockId), std::move(blockIdToVecIds) };
		assert(partition.blockIdToVecIds.size() == state.numClusters);

		return partition;
	}

} // namespace npq