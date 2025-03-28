#include <cassert>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <set>

#include "partition.h"


namespace npq
{

Partition::Partition(id_t numVecs, id_t numBlocks)
{
    vecIdToBlockId.resize(numVecs);
    blockIdToVecIds.resize(numBlocks);
}

Partition::Partition(std::vector<id_t>&& vecIdToBlockId, std::vector<std::vector<id_t>>&& blockIdToVecIds)
    : vecIdToBlockId{ std::move(vecIdToBlockId) }, blockIdToVecIds{ std::move(blockIdToVecIds) }
{}

bool Partition::operator==(const Partition& other) const
{
	// Replace vectors with sets to ignore order when comparing partitions
    std::set<std::set<id_t>> thisSet;
    for (const auto& vecIds : blockIdToVecIds)
    {
        std::set<id_t> vecIdsSet(vecIds.begin(), vecIds.end());
        thisSet.insert(vecIdsSet);
    }

    std::set<std::set<id_t>> otherSet;
    for (const auto& vecIds : other.blockIdToVecIds)
    {
        std::set<id_t> vecIdsSet(vecIds.begin(), vecIds.end());
        otherSet.insert(vecIdsSet);
    }

    return thisSet == otherSet;
}

Partition::Partition(Partition&& other) noexcept
   : vecIdToBlockId{ std::move(other.vecIdToBlockId) }, blockIdToVecIds{ std::move(other.blockIdToVecIds) }
{}

Partition& Partition::operator=(Partition&& other) noexcept
{
    vecIdToBlockId = std::move(other.vecIdToBlockId);
    blockIdToVecIds = std::move(other.blockIdToVecIds);

    return *this;
}

Partition Partition::copy() const
{
	// Copy the vector vecIdToBlockId
	std::vector<id_t> vecIdToBlockIdCopy{ vecIdToBlockId };

	// Deep copy the vector of vectors blockIdToVecIds
	std::vector<std::vector<id_t>> blockIdToVecIdsCopy;
	blockIdToVecIdsCopy.reserve(blockIdToVecIds.size());
	for (const auto& vecIds : blockIdToVecIds)
	{
		blockIdToVecIdsCopy.push_back(vecIds);
	}

	return Partition{ std::move(vecIdToBlockIdCopy), std::move(blockIdToVecIdsCopy) };
}


Partition jointPartition(const Partition& p, const Partition& q)
{
    // Compute the cartesian product of partition p and partition q:

    // The intersection of a block from p and a block from q is stored as prod[pBlockId][qBlockId]
    std::vector<std::unordered_map<id_t, std::vector<id_t>>> prod;
	prod.resize(p.blockIdToVecIds.size());
    id_t prodBlockCount = 0;

    // Iterate through the blocks in partition p
    for (id_t pBlockId = 0; pBlockId < p.blockIdToVecIds.size(); ++pBlockId)
    {
        // For each block in partition p, iterate through the vectors in the block
        for (const id_t& vecId : p.blockIdToVecIds[pBlockId])
        {
            // Find the block in partition q that each vector belongs to
            const id_t qBlockId = q.vecIdToBlockId[vecId];

            // Add the vector to the intersection of the current p and q blocks
            if (!prod[pBlockId].contains(qBlockId))
            {
                ++prodBlockCount;
            }
            prod[pBlockId][qBlockId].push_back(vecId);
        }
    }

    // Create the joint partition:

    id_t numVecs = p.vecIdToBlockId.size();
    Partition joint{ numVecs, prodBlockCount };
    id_t curJointBlockId = 0;
    // Iterate over all intersections in the cartesian product
    for (auto& pBlockProd : prod)
    {
        for (auto& [_, jointBlock] : pBlockProd)
        {
            // Update the map from vector id to block id
            for (const id_t& vecId : jointBlock)
            {
                joint.vecIdToBlockId[vecId] = curJointBlockId;
            }
            // Update the map from block id to vector ids
            joint.blockIdToVecIds[curJointBlockId] = std::move(jointBlock);

            ++curJointBlockId;
        }
    }
    
    return joint;
}


Partition jointPartitionByIndices(const std::vector<Partition>& partitions, const std::vector<dim_t>& indices)
{
	assert(!indices.empty());

	if (indices[0] == -1 && indices.size() == 1)
	{
        // Include all partitions in the joint
        Partition joint{ partitions[0].copy() };
		for (dim_t index = 1; index < partitions.size(); ++index)
		{
			joint = jointPartition(joint, partitions[index]);
		}

        return joint;
	}
    else
    {
        // Include only the specified partitions in the joint
        Partition joint{ partitions[indices[0]].copy() };
        for (size_t i = 1; i < indices.size(); ++i)
		{
            const dim_t index = indices[i];
			assert(index >= 0 && index < partitions.size());

			joint = jointPartition(joint, partitions[index]);
		}

        return joint;
    }
}


double entropy(const Partition& p, bool exp)
{
    const id_t numVecs = p.vecIdToBlockId.size();
    double acc;

    if (!exp)
    {
        // Compute entropy as the sum of -p log p
        acc = 0.0;
        for (const auto& block : p.blockIdToVecIds) {
            const double p = static_cast<double>(block.size()) / numVecs;
            acc -= p * std::log2(p);
        }
    }
    else
    {
        // Compute exponentiated entropy as the product of p ^ -p
        acc = 1.0;
        for (const auto& block : p.blockIdToVecIds) {
            const double p = static_cast<double>(block.size()) / numVecs;
            acc *= std::pow(p, -p);
        }
    }

    return acc;
}


//double JointEntropy(const Partition& p, const Partition& q, bool exp)
//{
//    return 0.0;
//}

} // namespace npq
