#include "graph.h"


namespace npq
{

	Graph::Graph(std::vector<std::vector<dim_t>>&& adj)
		: adj{ std::move(adj) }
	{
		// Extract edges from the adjacency list
		for (dim_t i = 0; i < this->adj.size(); ++i)
		{
			for (dim_t j : this->adj[i])
			{
				if (i < j)
				{
					edges.emplace_back(i, j);
				}
			}
		}
	}

}
