#pragma once

#include "parameters.h"
#include "partition.h"
#include "subspace_decomposition.h"


namespace npq
{

SubspaceDecomposition computeSubspaceDecompositionNaive(const std::vector<Partition>& partitions, const Parameters& params);

} // namespace npq