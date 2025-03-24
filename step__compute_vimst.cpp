/**
 * @brief The second step of the algorithm. See algorithm_steps.h for more information.
 */

#include <algorithm>

#include "algorithm_steps.h"
#include "partition.h"
#include "mst.h"


namespace npq
{

Graph computeVIMST(const std::vector<Partition>& partitions, const Parameters& params)
{
	const dim_t d = partitions.size();

	// Compute the entropy of the partition corresponding to each dimension
	std::vector<double> entropies;
	entropies.reserve(d);
	for (const Partition& p : partitions)
	{
		entropies.push_back(entropy(p));
	}

	// Compute the Variation of Information (VI) between each pair of partitions/dimensions
	std::vector<Edge> edges;
	edges.reserve(d * (d - 1) / 2);
	for (dim_t i = 0; i < d; ++i)
	{
		for (dim_t j = i + 1; j < d; ++j)
		{
			const double jointEntropy = entropy(jointPartition(partitions[i], partitions[j]));  // TODO: Implement JointEntropy
			const double variationOfInformation = 2 * jointEntropy - entropies[i] - entropies[j];

			edges.push_back({ variationOfInformation, i, j });
		}
	}

	// Compute the edges in the MST of the VI graph
	std::vector<Edge> mstEdges = kruskalMST(edges, d);

	// Construct the adjacency list representation of the MST
	std::vector<std::vector<dim_t>> mstAdj;
	mstAdj.resize(d);

	for (const Edge& e : mstEdges)
	{
		const dim_t i = std::get<1>(e);
		const dim_t j = std::get<2>(e);
		mstAdj[i].push_back(j);
		mstAdj[j].push_back(i);
	}
	// Note that by construction, the adjacency lists are sorted in non-decreasing order of edge weights
	// This must be true for the following call to the raviDCMST function to work correctly
	
	// Compute a degree-constrained approximation of the MST
	dim_t maxDegree = params.mstMaxDegree;
	if (maxDegree == -1)
	{
		maxDegree = static_cast<dim_t>(ceil(log2(d)));
	}
	raviDCMST(mstAdj, maxDegree);

	return Graph(std::move(mstAdj));
}

}