#include "dataset.h"


namespace npq
{

Dataset::Dataset(float* data, id_t n, dim_t d) {
	// Allocate memory
	dimensions.resize(d);
	for (dim_t dimId = 0; dimId < d; ++dimId) {
		dimensions[dimId].reserve(n);
	}

	// Loop over the row-major order array, where the index is i = vecId * d + dimId
	for (id_t vecId = 0; vecId < n; ++vecId) {
		for (dim_t dimId = 0; dimId < d; ++dimId) {
			dimensions[dimId].push_back(data[vecId * d + dimId]);
		}
	}
}

} // namespace npq