#pragma once

#include <vector>


/**
 * @brief The main function of the algorithm that performs natural product quantization.
 *
 * This function is a wrapper around the internal implementation of the algorithm, which is split into four steps:
 * 1. Compute 1D partitions
 * 2. Compute constraint tree
 * 3. Compute subspace decomposition
 * 4. Compute codebook
 *
 * @param data The input vector dataset, stored in row-major order (i.e. concatenated vectors, see dataset.h).
 * @param n The number of vectors in the subsampled dataset pointed to by 'data'.
 * @param d The dimensionality of each vector.
 * @param trueNumVectors The true number of vectors in the full dataset.
 * @param targetDistortion The target quantization distortion to achieve (see parameters.h).
 * @param targetDistortionMargin The margin of error within which NPQ achieves the target distortion (see parameters.h).
 * @param mstMaxDegree The maximum degree of each vertex of the constraint tree (see parameters.h).
 * @return The output codebook (and implicitly the subspace decomposition) of NPQ (see codebook.h).
 */
std::vector<std::pair<std::vector<short>, std::vector<std::vector<float>>>>
doNaturalProductQuantization(const float* data, int n, short d,
							 int trueNumVectors, double targetDistortion, double targetDistortionMargin = 0.0, short mstMaxDegree = 0);


/**
 * @brief The main function of an alternative simplified algorithm for testing and experimentation purposes.
 * 
 * @param data The input vector dataset, stored in row-major order (i.e. concatenated vectors, see dataset.h).
 * @param n The number of vectors in the subsampled dataset pointed to by 'data'.
 * @param d The dimensionality of each vector.
 * @param trueNumVectors The true number of vectors in the full dataset.
 * @param targetDistortion The target quantization distortion to achieve (see parameters.h).
 * @return The output subspace decomposition as a list of sets of dimensions,
 *         along with the recommended number of codewords for each subspace.
 */
std::vector<std::pair<std::vector<short>, int>> doNaiveNPQ(const float* data, int n, short d, int trueNumVectors, double targetDistortion);


/**
 * @brief Latest and greatest variation combining step 1 of the main algorithm with naive greedy merging.
 *
 * @param data The input vector dataset, stored in row-major order (i.e. concatenated vectors, see dataset.h).
 * @param n The number of vectors in the subsampled dataset pointed to by 'data'.
 * @param d The dimensionality of each vector.
 * @param trueNumVectors The true number of vectors in the full dataset.
 * @param targetDistortion The target quantization distortion to achieve (see parameters.h).
 * @param targetDistortionMargin The margin of error within which NPQ achieves the target distortion (see parameters.h).
 * @return The output codebook (and implicitly the subspace decomposition) of NPQ (see codebook.h).
 */
std::vector<std::pair<std::vector<short>, std::vector<std::vector<float>>>> doNotSoNaiveNPQ(const float* data, int n, short d, int trueNumVectors, double targetDistortion, double targetDistortionMargin = 0.0);
