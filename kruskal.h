#pragma once

#include "dataset.h"


namespace npq
{

/**
 * @brief Represents an edge of a weighted graph, where the vertex set is the set of dimensions.
 */
using Edge = std::tuple<const double, const dim_t, const dim_t>;

/**
 * @brief Computes the Minimum Spanning Tree (MST) of the given graph.
 *
 * This function implements Kruskal's algorithm to find the MST of the given graph.
 * Note that this function may reorder the input list of edges.
 *
 * @param edges The list of edges in the graph, where each edge is a tuple of the form (weight, u, v).
 * @return The MST of the graph.
 */
std::vector<Edge> KruskalMST(std::vector<Edge>& edges, dim_t d);

} // namespace npq