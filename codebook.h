#pragma once

#include <vector>

#include "dataset.h"


namespace npq
{

/**
 * @brief Represents a single cluster in a subspace as a vector of scalar components.
 * 
 * The i-th component of a Codeword corresponds to the i-th dimension of the associated Subspace.
 */
using Codeword = std::vector<scalar_t>;

/**
 * @brief Represents a subspace of the vector space as a list of dimension IDs.
 *
 * The ordering of the dimensions in this list corresponds to the ordering of the components of all
 * associated Codewords in a Subcodebook. Note that the dimensions are not necessarily contiguous or sorted.
 */
using Subspace = std::vector<dim_t>;

/**
 * @brief A pair of a subspace and a list of the codewords in that subspace.
 * 
 * The ordering of dimensions in the Subspace corresponds to the ordering of components in the Codewords.
 */
using Subcodebook = std::pair<Subspace, std::vector<Codeword>>;

/**
 * @brief A list of all sub-codebooks, one for each subspace in a subspace decomposition.
 * 
 * Each Subcodebook lists the dimensions of the subspace and the codewords in that subspace.
 */
using Codebook = std::vector<Subcodebook>;

} // namespace npq

