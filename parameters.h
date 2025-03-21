#pragma once

#include "dataset.h"


namespace npq
{

struct Parameters
{
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
	 * This margin is a small positive value that is subtracted from the target distortion to get the target total mean
	 * squared error for the separate 1D k-means clustering/partitioning of dimensions. This is done to ensure that the
	 * final optimal quantization computed by NPQ will achieve the target distortion with high probability.
	 * 
	 * The default value is <KNARF>.
	 */
	double targetDistortionMargin;

	/**
	 * @brief The maximum degree of each vertex in the degree-constrained approximation of the Minimum Spanning Tree (MST)
	 * used in the second step of the algorithm.
	 *
	 * The default value is ceil(log2(d)), where d is the number of dimensions in the dataset.
	 */
	dim_t mstMaxDegree;

	Parameters(double targetDistortion, double targetDistortionMargin = 0.0, dim_t mstMaxDegree = -1);
};

} // namespace npq