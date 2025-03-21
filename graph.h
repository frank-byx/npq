#pragma once

#include "Partition.h"


namespace npq
{

/**
 * @brief Represents a graph on the set of dimensions the dataset.
 */
struct Graph
{
	/**
	 * @brief The adjacency list of the graph.
	 *
	 * The adjacency list of the graph, where adj[i] is the list of neighbors of vertex i.
	 */
	std::vector<std::vector<dim_t>> adj;

	/**
	 * @brief Constructs a graph with vertices from 0 to d-1 and no edges.
	 *
	 * @param d The number of vertices/dimensionality of the dataset.
	 */
	Graph(dim_t d);
};

} // namespace npq