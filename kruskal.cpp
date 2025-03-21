#include <algorithm>
#include <numeric>
#include <vector>

#include "kruskal.h"


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


std::vector<Edge> KruskalMST(std::vector<Edge>& edges, dim_t d)
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

} // namespace npq