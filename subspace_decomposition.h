#pragma once

#include <vector>

#include "dataset.h"


namespace npq
{

/**
 * @brief Represents a decomposition of the set of dimensions of the dataset into subspaces.
 *
 * This struct holds two mappings that define how the dimensions are grouped into subspaces. It contains a mapping
 * from dimension IDs to the ID of the subspace that contains them, as well as a mapping from subspace IDs to the
 * IDs of the dimensions contained within those subspaces. The ID of a subspace is always the smallest dimension ID
 * out of the dimensions that the subspace contains.
 */
class SubspaceDecomposition
{
public:
	/**
	 * @brief Constructor that sets the number of dimensions and initializes the decomposition
	 * to either singleton subspaces each containing one dimension, or a single subspace containing all dimensions.
	 *
	 * @param numDims The dimensionality of the dataset.
	 * @param initSingletons If true, the decomposition is initialized to singleton subspaces.
	 * Otherwise, it is initialized to a single subspace containing all dimensions.
	 */
	SubspaceDecomposition(dim_t numDims, bool initSingletons);

	/**
	 * @brief Returns the number of dimensions in the dataset.
	 *
	 * @return The dimensionality of the dataset.
	 */
	dim_t numDims() const;

	/**
	 * @brief Returns the ID of the subspace that contains the given dimension.
	 * 
	 * The ID of a subspace is always the smallest dimension ID out of the dimensions that the subspace contains.
	 *
	 * @param dimId The ID of the dimension to query.
	 * @return The ID of the subspace that contains the given dimension.
	 */
	dim_t getSubspaceId(dim_t dimId) const;

	/**
	 * @brief Returns the IDs of the dimensions contained by the given subspace.
	 *
	 * @param subspaceId The ID of the subspace to query.
	 * @return A vector containing the IDs of the dimensions contained by the given subspace, in sorted order.
	 */
	std::vector<dim_t> getDimIds(dim_t subspaceId) const;

	/**
	 * @brief Returns the IDs of all subspaces in the decomposition.
	 *
	 * @return A vector containing the IDs of all subspaces in the decomposition, in sorted order.
	 */
	std::vector<dim_t> allSubspaceIds() const;

	/**
	 * @brief Checks if two dimensions are contained in the same subspace.
	 * 
	 * The two dimensions being compared must be distinct.
	 *
	 * @param dimId1 The ID of the first dimension to compare.
	 * @param dimId2 The ID of the second dimension to compare.
	 * @return true if the two dimensions are contained in the same subspace, false otherwise.
	 */
	bool inSameSubspace(dim_t dimId1, dim_t dimId2) const;

	/**
	 * @brief Merges two subspaces into a single subspace.
	 *
	 * Each subspace is identified by any single dimension that it contains. The two subspaces must be distinct.
	 * The ID of the resulting subspace is the smallest out of the IDs of the merged subspaces.
	 *
	 * @param dimId1 The ID of a dimension contained by the first subspace to merge.
	 * @param dimId2 The ID of a dimension contained by the second subspace to merge.
	 */
	void merge(dim_t dimId1, dim_t dimId2);

	/**
	 * @brief Helper function for SubspaceDecomposition::split that checks if a split of a subspace is valid.
	 *
	 * @param dimIds1 The IDs of the dimensions contained by the first subspace after the split.
	 * @param dimIds2 The IDs of the dimensions contained by the second subspace after the split.
	 * @return true if the split is valid, false otherwise.
	 */
	bool isValidSplit(const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2) const;

	/**
	 * @brief Splits a subspace into two subspaces.
	 *
	 * The split is described by a partition of the dimensions contained by the original subspace, given as a pair
	 * of disjoint sets of dimensions whose union is exactly the set of dimensions in the original subspace.
	 *
	 * @param dimIds1 The IDs of the dimensions contained by the first subspace after the split.
	 * @param dimIds2 The IDs of the dimensions contained by the second subspace after the split.
	 */
	void split(const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2);

	/**
	 * @brief Equality operator to check if two subspace decompositions are equivalent.
	 *
	 * @param other The other SubspaceDecomposition object to compare with.
	 * @return true if the two decompositions are equivalent, false otherwise.
	 */
	bool operator==(const SubspaceDecomposition& other) const;

private:
	/**
	 * @brief A map from each dimension to the subspace it belongs to.
	 *
	 * Implemented as an array where the index corresponds to a dimension ID and the value is the subspace ID.
	 * The ID of a subspace is always the smallest dimension ID out of the dimensions that the subspace contains.
	 */
	std::vector<dim_t> dimIdToSubspaceId;
};

} // namespace npq