#pragma once

#include <vector>


namespace npq
{

/**
 * @brief Defines a type alias for 32-bit signed integers used as IDs of vectors and any subsets or counts of vectors.
 *
 * It is assumed that the # of vectors in a dataset is n <= 2^31, and that IDs are from 0 to n-1.
 */
typedef int32_t id_t;


/**
 * @brief Defines a type alias for 16-bit signed integers used to label/count the dimensions of vectors in a dataset.
 *
 * It is assumed that the dimensionality of each vector is d <= 2^15, and that labels are from 0 to d-1.
 */
typedef int16_t dim_t;

/**
 * @brief Defines a type alias for 32-bit floating point numbers used to store each component of a vector in a dataset.
 */
typedef float scalar_t;


/**
 * @brief Stores a dataset of vectors with the values of each dimension contiguous in memory.
 *
 * Since we will process each dimension separately, we store the values of each dimension in a separate std::vector,
 * where each std::vector is indexed by the vector ID from 0 to n-1. These std::vectors are stored in another
 * std::vector, indexed by the dimension from 0 to d-1.
 */
struct Dataset {
	/**
	 * @brief The dataset, stored such that dimensions[i][j] is the value of
	 * the i-th dimension/component of the j-th vector in the dataset.
	 */
	std::vector<std::vector<scalar_t>> dimensions;

	/**
	 * @brief Constructor that takes a dataset in row-major order.
	 * 
	 * In Row-major order, the dataset is stored as a 1D array where the values of each vector are contiguous in memory.
	 * Therefore, we basically need to uncollate the vectors into separate dimensions, keeping their original order.
	 *
	 * @param data The dataset pointer (row-major order).
	 * @param n The number of vectors in the dataset.
	 * @param d The dimensionality of each vector.
	 */
	Dataset(scalar_t* data, id_t n, dim_t d);
};

} // namespace npq