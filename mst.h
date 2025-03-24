#pragma once

#include "dataset.h"


namespace npq
{

/**
 * @brief Represents an edge of a weighted graph, where the vertex set is the set of dimensions.
 */
using Edge = std::tuple<double, dim_t, dim_t>;

/**
 * @brief Computes the Minimum Spanning Tree (MST) of the given graph.
 *
 * This function implements Kruskal's algorithm to find the MST of the given graph.
 * Note that this function may reorder the input list of edges.
 * Furthermore, the output list of edges is sorted in non-decreasing order of edge weights.
 *
 * @param edges The list of edges in the graph, where each edge is a tuple of the form (weight, u, v).
 * @param d The number of vertices in the graph (equal to the dimensionality of the dataset).
 * @return The edges in the MST of the graph, sorted in non-decreasing order of edge weights.
 */
std::vector<Edge> kruskalMST(std::vector<Edge>& edges, dim_t d);

/**
 * @brief Computes a degree-constrained approximation of the given Minimum Spanning Tree (MST).
 *
 * Given an MST of a graph with edge weights satisfying the triangle inequality, and assuming that the adjacency
 * lists representing the MST are sorted in non-decreasing order of edge weights, this function computes a
 * degree-constrained 2-approximation of the MST using the algorithm presented in section 5 of the paper:
 * 
 * Ravi, R., Marathe, M. V., Ravi, S. S., Rosenkrantz, D. J., & Hunt, H. B. (1993).
 * Many birds with one stone: Multi-objective approximation algorithms.
 * Proceedings of the Twenty-Fifth Annual ACM Symposium on Theory of Computing (STOC '93), 438–447.
 * Association for Computing Machinery. https://doi.org/10.1145/167088.167209
 *
 * @param mstAdj The MST of a graph, represented by an adjacency list, which this function modifies
 * into a degree-constrained approximation of its original value.
 */
void raviDCMST(std::vector<std::vector<dim_t>>& mstAdj, dim_t maxDegree);

} // namespace npq