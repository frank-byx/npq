#pragma once

#include <vector>

#include "dataset.h"
#include "partition.h"
#include "parameters.h"
#include "graph.h"
#include "subspace_decomposition.h"


namespace npq
{

/**
 * @brief The first step of the algorithm.
 * 
 * Computes a 1D partition for each dimension of the data. Each partition represents a globally-optimal
 * 1D k-means clustering of the dataset with respect to the values of a single dimension.
 * 
 * @param dataset The processed dataset.
 * @param params Dictionary of input parameters.
 * 
 * @return A vector of partitions of the dataset, one for every dimension.
 */
std::vector<Partition> compute1dPartitions(const Dataset& dataset, const Parameters& params);

/**
 * @brief The second step of the algorithm.
 *
 * Constructs a complete weighted graph on the set of dimensions, where each node is a dimension and
 * the weight of each edge is the Variation of Information (VI) between the partitions corresponding to
 * the incident dimensions, and computes the Minimum Spanning Tree (MST) of this graph. The MST may then be
 * modified to satisfy a maximum degree constraint in order to improve the efficiency of the algorithm.
 *
 * @param partitions The 1D partitions of the dataset.
 * @param params Dictionary of input parameters.
 *
 * @return The (possibly degree-constrained approximation of the) MST of the VI graph over the dimensions.
 */
Graph computeVIMST(const std::vector<Partition>& partitions, const Parameters& params);

/**
 * @brief The third step of the algorithm.
 * 
 * Defines an objective function that minimizes the total memory footprint of the codebooks and vector storage over the
 * set of partitions of the set of dimensions, i.e. subspace decompositions. This objective function is optimized under
 * the constraint that each subspace must correspond to the vertex set of a connected subtree of the given tree.
 * The optimization algorithm is a greedy algorithm that iteratively splits and merges subspaces on edges in the tree.
 * 
 * @param partitions The 1D partitions of the dataset.
 * @param tree The tree that constrains the search over subspace decompositions.
 * @param params Dictionary of input parameters.
 * 
 * @return The most optimal subspace decomposition found by the greedy algorithm.
 */
SubspaceDecomposition computeSubspaceDecomposition(
	const std::vector<Partition>& partitions,
	const Graph& tree,
	const Parameters& params
);

} // namespace npq