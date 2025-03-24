#pragma once

#include "dataset.h"


namespace npq
{

/**
 * @brief Represents a graph on the set of dimensions of the dataset.
 */
struct Graph
{
	/**
	 * @brief The adjacency list of the graph.
	 */
	std::vector<std::vector<dim_t>> adj;

	/**
	 * @brief The edges of the graph.
	 *
	 * For each edge, the first dimension ID is less than the second dimension ID.
	 */
	std::vector<std::pair<dim_t, dim_t>> edges;

	/**
	 * @brief Constructor to initialize the graph by moving in the given adjacency list and extracting the edges.
	 *
	 * @param adj The adjacency list of the graph (rvalue reference).
	 */
	Graph(std::vector<std::vector<dim_t>>&& adj);
};

} // namespace npq