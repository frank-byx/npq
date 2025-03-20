#pragma once

#include <vector>
#include "dataset.h"


namespace npq
{

/**
 * @brief Represents a partition of a set of vectors into blocks.
 *
 * This struct holds two mappings that define how a set of vectors (identified by IDs) are grouped into blocks.
 * It contains a mapping from vector IDs to the block ID that contains them, as well as a mapping from
 * block IDs to the vector IDs contained within those blocks.
 */
struct Partition
{
    /**
	 * @brief A map that associates each vector ID to the block ID it belongs to.
	 *
	 * This vector serves as a lookup table, where the index corresponds to a vector ID,
	 * and the value at that index represents the ID of the block containing that vector.
	 */
	std::vector<id_t> vecIdToBlockId;

	/**
	 * @brief A map that associates each block ID to a list of vector IDs contained within that block.
	 *
	 * This vector of vectors serves as a lookup table, where the outer index corresponds to a block ID,
	 * and the value at that index is a list of the vector IDs that belong to the specified block.
	 * This allows for efficient look-up of all vectors contained within a particular block.
	 */
	std::vector<std::vector<id_t>> blockIdToVecIds;
	
	/**
	 * @brief Constructor to initialize the maps to the given sizes.
	 *
	 * @param numVecs The size for the vecIdToBlockId vector.
	 * @param numBlocks The size for the blockIdToVecIds (outer) vector.
	 */
	Partition(id_t numVecs = 0, id_t numBlocks = 0);

	/**
	 * @brief Constructor to directly set the maps from rvalues.
	 *
	 * @param vecIdToBlockId The vecIdToBlockId vector (rvalue reference).
	 * @param blockIdToVecIds The blockIdToVecIds (outer) vector (rvalue reference).
	 */
	Partition(std::vector<id_t>&& vecIdToBlockId, std::vector<std::vector<id_t>>&& blockIdToVecIds);

	/**
	 * @brief Equality operator to compare two partitions for testing purposes.
	 *
	 * This operator compares two Partition objects to check if the partitions they represent are equal
	 * by replacing std::vectors with std::sets to ignore order. This is only called from unit tests.
	 *
	 * @param other The Partition object to compare with the current object.
	 * @return Returns true if both Partition objects are equal (i.e., their
	 *         sets of sets are identical), otherwise returns false.
	 */
	bool operator==(const Partition& other) const;
};


/**
 * @brief Compute the joint partition of two partitions.
 *
 * The joint partition refers to the Cartesian product partition of the two given partitions, where each block in the
 * resulting partition is the set intersection of a block from the first given partition and a block from the second.
 * The order of arguments does not matter.
 *
 * @param p The first given partition.
 * @param q The second given partition.
 *
 * @return The computed joint partition.
 */
Partition JointPartition(const Partition& p, const Partition& q);


/**
 * @brief Compute the entropy of the given partition.
 *
 * This function calculates the empirical entropy of the given partition of vectors. The logarithm used is base 2,
 * for Shannon entropy. Optionally, it can return the exponential (base 2) of the entropy.
 *
 * @param p The given partition.
 * @param exp If true, return the exponentiated entropy. Default is false.
 *
 * @return The computed entropy (or exponentiated entropy if exp is true).
 */
double Entropy(const Partition& p, bool exp = false);


// TODO: Add optimized composite functions after everything else works

/**
 * @brief Compute the entropy of the joint partition of two partitions.
 *
 * This function calculates the empirical entropy of the joint partition of the two given partitions of vectors.
 * It should return the same result as Entropy(JointPartition(p, q)), but more efficiently by not storing the
 * intermediate joint partition. It can also optionally return the exponential of the entropy.
 *
 * @param p The first given partition.
 * @param q The second given partition.
 * @param exp If true, return the exponentiated entropy. Default is false.
 *
 * @return The computed entropy (or exponentiated entropy if exp is true).
 */
//double JointEntropy(const Partition& p, const Partition& q, bool exp = false);

} // namespace npq