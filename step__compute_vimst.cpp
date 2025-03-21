/**
 * @brief The second step of the algorithm. See algorithm_steps.h for more information.
 */

#include <algorithm>

#include "algorithm_steps.h"
#include "partition.h"
#include "kruskal.h"


namespace npq
{

Graph ComputeVIMST(std::vector<Partition> partitions)
{
	const dim_t d = partitions.size();

	// Compute the entropy of the partition corresponding to each dimension
	std::vector<double> entropy;
	entropy.reserve(d);
	for (const Partition& p : partitions)
	{
		entropy.push_back(Entropy(p));
	}

	// Compute the Variation of Information (VI) between each pair of partitions/dimensions
	std::vector<Edge> edges;
	edges.reserve(d * (d - 1) / 2);
	for (dim_t i = 0; i < d; ++i)
	{
		for (dim_t j = i + 1; j < d; ++j)
		{
			const double jointEntropy = Entropy(JointPartition(partitions[i], partitions[j]));  // TODO: Implement JointEntropy
			const double variationOfInformation = 2 * jointEntropy - entropy[i] - entropy[j];

			edges.push_back({ variationOfInformation, i, j });
		}
	}

	// Compute the edges in the MST of the VI graph
	std::vector<Edge> mstEdges = KruskalMST(edges, d);

	// Construct the MST in the form of an adjacency list
	Graph vimst(d);
	for (const Edge& e : mstEdges)
	{
		const dim_t i = std::get<1>(e);
		const dim_t j = std::get<2>(e);
		vimst.adj[i].push_back(j);
		vimst.adj[j].push_back(i);
	}
	
	return vimst;
}

}