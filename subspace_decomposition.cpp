#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>

#include <iostream>

#include "subspace_decomposition.h"


namespace npq
{

SubspaceDecomposition::SubspaceDecomposition(std::vector<dim_t>&& dimIdToSubspaceId)
	: dimIdToSubspaceId(std::move(dimIdToSubspaceId)) {}

SubspaceDecomposition::SubspaceDecomposition(dim_t numDims, bool initSingletons) : dimIdToSubspaceId(numDims, 0)
{
	if (initSingletons)
	{
		std::iota(dimIdToSubspaceId.begin(), dimIdToSubspaceId.end(), 0);
	}
}

dim_t SubspaceDecomposition::numDims() const
{
	return dimIdToSubspaceId.size();
}

dim_t SubspaceDecomposition::getSubspaceId(dim_t dimId) const
{
	return dimIdToSubspaceId[dimId];
}

std::vector<dim_t> SubspaceDecomposition::getDimIds(dim_t subspaceId) const
{
	std::vector<dim_t> dimIds;
	for (dim_t i = 0; i < numDims(); ++i)
	{
		if (dimIdToSubspaceId[i] == subspaceId)
		{
			dimIds.push_back(i);
		}
	}

	return dimIds;
}

std::vector<dim_t> SubspaceDecomposition::allSubspaceIds() const
{
	std::vector<dim_t> subspaceIds = dimIdToSubspaceId;
	std::sort(subspaceIds.begin(), subspaceIds.end());
	subspaceIds.erase(std::unique(subspaceIds.begin(), subspaceIds.end()), subspaceIds.end());

	return subspaceIds;
}

bool SubspaceDecomposition::inSameSubspace(dim_t dimId1, dim_t dimId2) const
{
	assert(dimId1 != dimId2);

	return dimIdToSubspaceId[dimId1] == dimIdToSubspaceId[dimId2];
}

void SubspaceDecomposition::merge(dim_t dimId1, dim_t dimId2)
{
	assert(!inSameSubspace(dimId1, dimId2));

	if (dimIdToSubspaceId[dimId1] > dimIdToSubspaceId[dimId2])
	{
		std::swap(dimId1, dimId2);
	}

	const dim_t newSubspaceId = dimIdToSubspaceId[dimId1];
	const dim_t oldSubspaceId = dimIdToSubspaceId[dimId2];
	assert(newSubspaceId < oldSubspaceId);

	for (dim_t i = 0; i < numDims(); ++i)
	{
		if (dimIdToSubspaceId[i] == oldSubspaceId)
		{
			dimIdToSubspaceId[i] = newSubspaceId;
		}
	}
}

bool SubspaceDecomposition::isValidSplit(const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2) const
{
	const std::vector<dim_t> oldSubspace = getDimIds(dimIdToSubspaceId[dimIds1[0]]);
	const std::set<dim_t> oldSubspaceSet = std::set<dim_t>(oldSubspace.begin(), oldSubspace.end());
	assert(oldSubspaceSet.size() == oldSubspace.size());
		
	const std::set<dim_t> dimIds1Set(dimIds1.begin(), dimIds1.end());
	const std::set<dim_t> dimIds2Set(dimIds2.begin(), dimIds2.end());
	if (dimIds1Set.size() != dimIds1.size() || dimIds2Set.size() != dimIds2.size())
	{
		return false;
	}

	for (dim_t dimId = 0; dimId < numDims(); ++dimId)
	{
		if (oldSubspaceSet.count(dimId))
		{
			if ((dimIds1Set.count(dimId) && dimIds2Set.count(dimId)) ||
				(!dimIds1Set.count(dimId) && !dimIds2Set.count(dimId)))
			{
				return false;
			}
		}
		else
		{
			if (dimIds1Set.count(dimId) || dimIds2Set.count(dimId))
			{
				return false;
			}
		}
	}

	return true;
}

void SubspaceDecomposition::split(const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2)
{
	assert(isValidSplit(dimIds1, dimIds2));

	const dim_t newSubspaceId1 = *std::min_element(dimIds1.begin(), dimIds1.end());
	const dim_t newSubspaceId2 = *std::min_element(dimIds2.begin(), dimIds2.end());

	for (const dim_t& dimId : dimIds1)
	{
		dimIdToSubspaceId[dimId] = newSubspaceId1;
	}
	for (const dim_t& dimId : dimIds2)
	{
		dimIdToSubspaceId[dimId] = newSubspaceId2;
	}
}

bool SubspaceDecomposition::operator==(const SubspaceDecomposition& other) const
{
	return dimIdToSubspaceId == other.dimIdToSubspaceId;
}

} // namespace npq