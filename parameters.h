#pragma once

#include "dataset.h"


namespace npq
{

struct Parameters
{
	/**
	 * @brief The true number of vectors in the dataset.
	 *
	 * The number of vectors in the full dataset, and not the number of vectors in the subsampled set that this
	 * algorithm takes as input. This is used to compute the cost of a subspace during step 3 of the algorithm.
	 */
	id_t trueNumVectors;
	
	/**
	 * @brief The target quantization distortion to achieve, set by the user.
	 *
	 * Quantization distortion is defined as the mean squared error of the quantized vectors from the original vectors.
	 * The intial separate 1D k-means clustering/partitioning of dimensions will choose (possibly varying) k such that
	 * the sum of the mean squared errors of all clusterings/dimensions is less than this target by a certain margin,
	 * so that the final optimal quantization computed by NPQ will achieve this target with high probability.
	 */
	double targetDistortion;

	/**
	 * @brief The margin of error within which NPQ maintains the quantization distortion as an approximate invaraiant.
	 *
	 * This margin is a small fractional value in [0, 1) that is subtracted from 1 and then multiplied by the target
	 * quantization distortion to get the target total mean squared error for the 1D k-means clustering step. This is
	 * to ensure that the final quantization computed by NPQ will achieve the target distortion with high probability.
	 * For example, if this is set to 0.02, the target MSE used by the algorithm will be 0.98 * targetDistortion.
	 * 
	 * The default value is 0.
	 */
	double targetDistortionMargin;

	/**
	 * @brief The maximum degree of each vertex in the degree-constrained approximation of the Minimum Spanning Tree (MST)
	 * used in the second step of the algorithm. The runtime complexity of the algorithm is linear in this parameter.
	 *
	 * The default value of 0 indicates to use max(2, ceil(log2(d))), where d is the number of dimensions in the dataset.
	 * A value of -1 indicates to not constrain the maximum degree of the MST, effectively setting the maximum to d-1.
	 */
	dim_t mstMaxDegree;

	Parameters(id_t trueNumVectors, double targetDistortion, double targetDistortionMargin = 0.0, dim_t mstMaxDegree = 0);
};

} // namespace npq