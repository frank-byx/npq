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


Partition JointPartition(const Partition& p, const Partition& q)
{
    // Compute the cartesian product of partition p and partition q:

    // The intersection of a block from p and a block from q is stored as prod[pBlockId][qBlockId]
    std::vector<std::unordered_map<id_t, std::vector<id_t>>> prod{ p.blockIdToVecIds.size() };
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
                prod[pBlockId][qBlockId] = std::vector<id_t>{};
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


double Entropy(const Partition& p, bool exp)
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
