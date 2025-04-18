/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>


namespace faiss {

	typedef int idx_t;  // In FAISS this typedef is of int64_t, but we use int32 instead

    /** SMAWK algorithm. Find the row minima of a monotone matrix.
     *
     * Expose this for testing.
     *
     * @param nrows    number of rows
     * @param ncols    number of columns
     * @param x        input matrix, size (nrows, ncols)
     * @param argmins  argmin of each row
     */
    void smawk(
        const idx_t nrows,
        const idx_t ncols,
        const float* x,
        idx_t* argmins);

    /** Exact 1D K-Means by dynamic programming
     *
     * From  "Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D"
     * Allan Gr�nlund, Kasper Green Larsen, Alexander Mathiasen, Jesper Sindahl
     * Nielsen, Stefan Schneider, Mingzhou Song, ArXiV'17
     *
     * Section 2.2
     *
     * https://arxiv.org/abs/1701.07204
     *
     * @param x          input 1D array
     * @param n          input array length
     * @param nclusters  number of clusters
     * @param centroids  output centroids, size nclusters
     * @return  imbalancce factor
     */
    double kmeans1d(const float* x, size_t n, size_t nclusters, float* centroids);

} // namespace faiss