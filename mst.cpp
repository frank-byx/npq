#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "mst.h"


namespace npq
{

/**
 * @brief Disjoint set data structure for Kruskal's algorithm over the set of dimensions.
 */
class DisjointSet
{
public:
    DisjointSet(dim_t d)
    {
		parent.resize(d);
        std::iota(parent.begin(), parent.end(), 0);

		rank.resize(d, 0);
    }

    dim_t find(dim_t i)
    {
        if (i == parent[i])
        {
			return i;
        }

        return parent[i] = find(parent[i]);
    }

    void merge(dim_t i, dim_t j)
    {
        i = find(i);
        j = find(j);

        if (i == j)
        {
            return;
        }

        if (rank[i] < rank[j])
        {
			std::swap(i, j);
        }
        parent[j] = i;

        if (rank[i] == rank[j])
        {
            ++rank[i];
        }
    }

private:
    std::vector<dim_t> parent;
    std::vector<dim_t> rank;
};


std::vector<Edge> kruskalMST(std::vector<Edge>& edges, dim_t d)
{
    std::sort(edges.begin(), edges.end());
    
    std::vector<Edge> result;
	result.reserve(d - 1);

    DisjointSet ds(d);

    for (const Edge& e : edges)
    {
        const dim_t i = std::get<1>(e);
		const dim_t j = std::get<2>(e);
        if (ds.find(i) == ds.find(j))
        {
            continue;
        }

        ds.merge(i, j);

        result.push_back(e);
		if (result.size() == d - 1)
		{
			break;
		}
    }
    
    return result;
}


// DFS helper function for raviDCMST
void DFS(const dim_t curNode, const dim_t parentNode, std::vector<std::vector<dim_t>>& mstAdj, const dim_t maxDegree)
{
	// Base case - leaf node
	const dim_t curDegree = mstAdj[curNode].size();
    if (curDegree == 1)
    {
		return;
    }

    // Do shortcutting if necessary
	const dim_t shortcutsNeeded = curDegree - maxDegree;
	dim_t shortcutsDone = 0;
    dim_t i = 0;
	dim_t prevChild = -1;
	while (shortcutsDone < shortcutsNeeded)
	{
		const dim_t curChild = mstAdj[curNode][i];

        // Skip the parent
		if (curChild == parentNode)
		{
			++i;
			continue;
		}

		// Skip the first proper child
        if (prevChild == -1)
        {
            prevChild = curChild;

            ++i;
            continue;
        }

		// Remove the edge between the current node and the current child
        mstAdj[curNode].erase(mstAdj[curNode].begin() + i);
        mstAdj[curChild].erase(std::find(mstAdj[curChild].begin(), mstAdj[curChild].end(), curNode));

		// Add the edge between the previous child and the current child
        mstAdj[prevChild].push_back(curChild);
        mstAdj[curChild].push_back(prevChild);

		// Update the previous child
		prevChild = curChild;

		++shortcutsDone;
	}

    assert(mstAdj[curNode].size() <= maxDegree);

	// Recurse on the children
	for (const dim_t& child : mstAdj[curNode])
	{
		if (child == parentNode)
		{
			continue;
		}

		DFS(child, curNode, mstAdj, maxDegree);
	}
}

void raviDCMST(std::vector<std::vector<dim_t>>& mstAdj, dim_t maxDegree)
{
    assert(maxDegree >= 2);
	dim_t d = mstAdj.size();

	// Find the root of the tree
	dim_t root = -1;
	for (dim_t i = 0; i < d; ++i)
	{
        const dim_t degree = mstAdj[i].size();
		if (degree >= 2)
		{
			root = i;
			break;
		}
	}
	if (root == -1)
	{
		return;
	}

    DFS(root, -1, mstAdj, maxDegree);
}

} // namespace npq