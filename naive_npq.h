#pragma once

#include <vector>

#include "dataset.h"


namespace npq
{

/**
 * @brief Alternative simplified algorithm for Natural Product Quantization, for testing and experimentation purposes.
 *
 * This algorithm does not balance the entropies of the 1D partitions, it uses a simplified greedy merging algorithm
 * that doesn't use an MST heuristic, and it only returns a subspace decomposition without providing the codebooks.
 *
 * @param dataset The processed dataset.
 * @param targetDistortion The input target distortion.
 * @return The subspace decomposition computed by the algorithm, as a list of sets of dimensions.
 */
std::vector<std::vector<dim_t>> naiveNPQ(const Dataset& dataset, double targetDistortion);

} // namespace npq
