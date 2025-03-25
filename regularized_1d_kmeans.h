/*
 * The code below is adapated from the code used for the experiments in the paper:
 * Grønlund, A., Larsen, K. G., Mathiasen, A., Nielsen, J. S., Schneider, S., & Song, M. (2018).
 * Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D. arXiv. https://arxiv.org/abs/1701.07204
 * from Allan Grønlund's GitHub repository: https://github.com/gronlund/kmeans1d.
 */

#pragma once

#include "1d_kmeans.h"


namespace npq
{

/*
 * @brief Computes the optimal 1D regularized k-means clustering.
 * 
 * The algorithm is described in section 2.4 of the paper:
 * Grønlund, A., Larsen, K. G., Mathiasen, A., Nielsen, J. S., Schneider, S., & Song, M. (2018).
 * Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D. arXiv. https://arxiv.org/abs/1701.07204
 * 
 * @param lambda The regularization term.
 * @param cc The cluster cost function.
 * @param n The number of data points.
 * 
 * @return A list of k indices of the sorted data, each the leftmost element of one of k clusters, from left to right.
 * I.e., the clusters are intervals: [ret[0], ret[1]-1], [ret[1], ret[2]-1], ..., [ret[k-1], n-1], where ret[0] = 0.
 */
std::vector<id_t> computeRegularized1DKMeans(double lambda, const CostCalculator& cc, id_t n);

} // namespace npq