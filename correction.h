#pragma once

#include "partition.h"
#include "parameters.h"


namespace npq
{

double estimateNumClusters(double expEntropy, const std::vector<dim_t>& subspaceDimIds, const Parameters& params, const std::vector<Partition>& partitions);

} // namespace npq