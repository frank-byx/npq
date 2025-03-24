#pragma once

#include <map>

#include "graph.h"
#include "partition.h"
#include "subspace_decomposition.h"


namespace npq
{

/**
 * @brief Defines the cost of a subspace as a function of exponentiated joint entropy.
 *
 * We estimate the number of codewords required to represent the subspace within the target quantization
 * distortion by the exponential of the entropy of the joint partition of the 1D partitions corresponding to
 * the dimensions of the subspace. Then, the cost of the subspace is defined as the sum of the expected
 * memory footprints of the corresponding subvector storage and sub-codebook (in bits):
 *
 * Estimated # of codewords:	k (= expEntropy)
 * # of dimensions in subspace:	d_sub (= subspaceDims)
 * # of vectors in dataset:		n (= numVectors)
 * Size of vector scalar type:	c (= sizeof(scalar_t))  (scalar_t is defined in dataset.h)
 *
 * Subvector storage memory:	A = n * log2(k)
 * Sub-codebook memory:			B = k * d_sub * c
 *
 * Subspace cost:				cost = A + B
 *
 * Note that ceil(log2(k)) is the true exepected subvector code length, but we omit the ceil function when
 * estimating subvector storage memory in order to reduce the complexity of the optimization objective.
 *
 * @param expEntropy The exponentiated joint entropy.
 * @param subspaceDims The number of dimensions in the subspace.
 */
double subspaceCost(double expEntropy, dim_t subspaceDims, id_t numVectors);


/**
 * @brief Represention of a potential merge operation between two subspaces, to be queued by GreedyMerger.
 */
struct MergeOperation
{
	/**
	 * @brief The change in total cost of the decomposition if the merge is performed.
	 */
	double changeInTotalCost;

	/**
	 * @brief The edge in the tree over the dimensions corresponding to the merge.
	 */
	std::pair<dim_t, dim_t> edge;

	/**
	 * @brief The joint partition of the resulting subspace.
	 */
	Partition newPartition;

	/**
	 * @brief The cost of the resulting subspace.
	 */
	double newCost;

	/**
	 * @brief The less-than operator to be used by a min priority queue.
	 * 
	 * @param other The other MergeOperation to compare to.
	 */
	bool operator<(const MergeOperation& other) const;

	/**
	 * @brief Constructor to initialize the merge operation with the given values.
	 * 
	 * @param changeInTotalCost The change in total cost of the decomposition if the merge is performed.
	 * @param edge The edge in the tree over the dimensions corresponding to the merge.
	 * @param newPartition The joint partition of the resulting subspace.
	 * @param newCost The cost of the resulting subspace.
	 */
	MergeOperation(double changeInTotalCost,
				   const std::pair<dim_t, dim_t>& edge,
				   Partition&& newPartition,
				   double newCost);

	/**
	 * @brief Move constructor.
	 * 
	 * @param other The other MergeOperation to move from.
	 */
	MergeOperation(MergeOperation&& other) noexcept;

	/**
	 * @brief Move assignment operator.
	 * 
	 * @param other The other MergeOperation to move from.
	 * @return The current MergeOperation object after moving.
	 */
	MergeOperation& operator=(MergeOperation&& other) noexcept;
};

/**
 * @brief Greedy algorithm that iteratively merges subspaces.
 */
class GreedyMerger
{
public:
	/**
	 * @brief Constructor to initialize the algorithm with the given decomposition, tree, and 1D partitions.
	 * 
	 * Populates a priority queue of potential merges of pairs of subspaces connected by edges in the tree,
	 * ordered by lowest (most negative) change in total cost of the decomposition.
	 *
	 * @param decomp The subspace decomposition to be modified.
	 * @param tree The tree over the dimensions of the dataset.
	 * @param partitions The 1D partitions of the dataset.
	 */
	GreedyMerger(SubspaceDecomposition& decomp, const Graph& tree, const std::vector<Partition>& partitions);

	/**
	 * @brief Checks if there are any merges left to perform that decrease the total cost of the decomposition.
	 *
	 * @return true if there are merges left to perform, false otherwise.
	 */
	bool canMerge() const;

	/**
	 * @brief Performs the best available merge operation.
	 * 
	 * There must be an available merge operation that decreases the total cost of the decomposition.
	 */
	void doMerge();

	/**
	 * @brief Returns the history of the total cost of the decomposition after each merge operation.
	 *
	 * @return A list where the cost after the i-th merge operation is at index i, and the initial cost is at index 0.
	 */
	std::vector<double> getLosses() const;

private:
	SubspaceDecomposition* const pDecomp;
	const Graph* const pTree;
	const std::vector<Partition>* const pPartitions;

	std::map<dim_t, Partition> subspaceIdToJointPartition;
	std::map<dim_t, double> subspaceIdToCost;
	std::vector<MergeOperation> mergeQueue;  // TODO: For simplicity, we just implement the queue as an array

	std::vector<double> losses;
};


/**
 * @brief Represention of a potential split operation of a subspace into two, to be queued by GreedySplitter.
 */
struct SplitOperation
{
	/**
	 * @brief The change in total cost of the decomposition if the split is performed.
	 */
	double changeInTotalCost;

	/**
	 * @brief The edge in the tree over the dimensions corresponding to the split.
	 *
	 * NOTE: The first dimension of the edge may be greater or smaller than the second dimension.
	 */
	std::pair<dim_t, dim_t> edge;

	/**
	 * @brief The IDs of the dimensions contained by the first resulting subspace after the split,
	 * i.e. the subspace containing the first incident dimension of the edge corresponding to the split.
	 */
	std::vector<dim_t> dimIds1;

	/**
	 * @brief The IDs of the dimensions contained by the second resulting subspace after the split,
	 * i.e. the subspace containing the second incident dimension of the edge corresponding to the split.
	 */
	std::vector<dim_t> dimIds2;

	/**
	 * @brief The joint partition of the first resulting subspace.
	 */
	Partition newPartition1;

	/**
	 * @brief The joint partition of the second resulting subspace.
	 */
	Partition newPartition2;

	/**
	 * @brief The cost of the first resulting subspace.
	 */
	double newCost1;

	/**
	 * @brief The cost of the second resulting subspace.
	 */
	double newCost2;

	/**
	 * @brief The less-than operator to be used by a min priority queue.
	 *
	 * @param other The other SplitOperation to compare to.
	 */
	bool operator<(const SplitOperation& other) const;

	/**
	 * @brief Constructor to initialize the split operation with the given values.
	 *
	 * @param changeInTotalCost The change in total cost of the decomposition if the split is performed.
	 * @param edge The edge in the tree over the dimensions corresponding to the split.
	 * @param dimIds1 The IDs of the dimensions contained by the first resulting subspace after the split.
	 * @param dimIds2 The IDs of the dimensions contained by the second resulting subspace after the split.
	 * @param newPartition1 The joint partition of the first resulting subspace.
	 * @param newPartition2 The joint partition of the second resulting subspace.
	 * @param newCost1 The cost of the first resulting subspace.
	 * @param newCost2 The cost of the second resulting subspace.
	 */
	SplitOperation(double changeInTotalCost,
				   const std::pair<dim_t, dim_t>& edge,
				   const std::vector<dim_t>& dimIds1, const std::vector<dim_t>& dimIds2,
				   Partition&& newPartition1, Partition&& newPartition2,
				   double newCost1, double newCost2);

	/**
	 * @brief Move constructor.
	 *
	 * @param other The other SplitOperation to move from.
	 */
	SplitOperation(SplitOperation&& other) noexcept;

	/**
	 * @brief Move assignment operator.
	 *
	 * @param other The other SplitOperation to move from.
	 * @return The current SplitOperation object after moving.
	 */
	SplitOperation& operator=(SplitOperation&& other) noexcept;
};

class GreedySplitter
{
public:
	/**
	 * @brief Constructor to initialize the algorithm with the given decomposition, tree, and 1D partitions.
	 *
	 * Populates a priority queue of potential splits of subspaces on contained edges in the tree,
	 * ordered by lowest (most negative) change in total cost of the decomposition.
	 *
	 * @param decomp The subspace decomposition to be modified.
	 * @param tree The tree over the dimensions of the dataset.
	 * @param partitions The 1D partitions of the dataset.
	 */
	GreedySplitter(SubspaceDecomposition& decomp, const Graph& tree, const std::vector<Partition>& partitions);

	/**
	 * @brief Checks if there are any splits left to perform that decrease the total cost of the decomposition.
	 *
	 * @return true if there are splits left to perform, false otherwise.
	 */
	bool canSplit() const;

	/**
	 * @brief Performs the best available split operation.
	 *
	 * There must be an available split operation that decreases the total cost of the decomposition.
	 */
	void doSplit();

	/**
	 * @brief Returns the history of the total cost of the decomposition after each split operation.
	 *
	 * @return A list where the cost after the i-th split operation is at index i, and the initial cost is at index 0.
	 */
	std::vector<double> getLosses() const;

private:
	/**
	 * @brief Recursive helper function.
	 */
	void postorderDFS(dim_t curDimId, dim_t parentDimId,
					  std::map<dim_t, Partition>& subtreeJointPartitions,
					  std::map<dim_t, std::vector<dim_t>>& subtreeDimIds,
					  std::map<dim_t, dim_t>& childToParentDimId);

	/**
	 * @brief Recursive helper function.
	 */
	void preorderDFS(dim_t curDimId, dim_t parentDimId,
					 std::map<dim_t, Partition>& complementJointPartitions,
					 std::map<dim_t, std::vector<dim_t>>& complementDimIds,
					 const std::map<dim_t, Partition>& subtreeJointPartitions,
					 const std::map<dim_t, std::vector<dim_t>>& subtreeDimIds);

	SubspaceDecomposition* const pDecomp;
	const Graph* const pTree;
	const std::vector<Partition>* const pPartitions;

	std::map<dim_t, double> subspaceIdToCost;
	std::vector<SplitOperation> splitQueue;  // TODO: For simplicity, we just implement the queue as an array

	std::vector<double> losses;
};

} // namespace npq