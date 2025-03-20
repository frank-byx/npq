/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * The code below is adapated from code from the FAISS repository licensed under the MIT license
 * found in the LICENSE file in the root directory of the FAISS repository.
 */

#include <functional>
#include <numeric>
#include <unordered_map>

#include "smawk.h"


namespace npq
{

void reduce(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const std::function<float(id_t, id_t)>& lookup,
    std::vector<id_t>& output_cols) {
    for (id_t col : input_cols) {
        while (!output_cols.empty()) {
            id_t row = rows[output_cols.size() - 1];
            float a = lookup(row, col);
            float b = lookup(row, output_cols.back());
            if (a >= b) { // defeated
                break;
            }
            output_cols.pop_back();
        }
        if (output_cols.size() < rows.size()) {
            output_cols.push_back(col);
        }
    }
}

void interpolate(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& cols,
    const std::function<float(id_t, id_t)>& lookup,
    id_t* argmins) {
    std::unordered_map<id_t, id_t> id_to_col;
    for (id_t idx = 0; idx < cols.size(); ++idx) {
        id_to_col[cols[idx]] = idx;
    }

    id_t start = 0;
    for (id_t r = 0; r < rows.size(); r += 2) {
        id_t row = rows[r];
        id_t end = cols.size() - 1;
        if (r < rows.size() - 1) {
            id_t idx = argmins[rows[r + 1]];
            end = id_to_col[idx];
        }
        id_t argmin = cols[start];
        float min = lookup(row, argmin);
        for (id_t c = start + 1; c <= end; c++) {
            float value = lookup(row, cols[c]);
            if (value < min) {
                argmin = cols[c];
                min = value;
            }
        }
        argmins[row] = argmin;
        start = end;
    }
}

/** SMAWK algo. Find the row minima of a monotone matrix.
    *
    * References:
    *   1. http://web.cs.unlv.edu/larmore/Courses/CSC477/monge.pdf
    *   2. https://gist.github.com/dstein64/8e94a6a25efc1335657e910ff525f405
    *   3. https://github.com/dstein64/kmeans1d
    */
void smawk_impl(
    const std::vector<id_t>& rows,
    const std::vector<id_t>& input_cols,
    const std::function<float(id_t, id_t)>& lookup,
    id_t* argmins) {
    if (rows.size() == 0) {
        return;
    }

    /**********************************
        * REDUCE
        **********************************/
    auto ptr = &input_cols;
    std::vector<id_t> survived_cols; // survived columns
    if (rows.size() < input_cols.size()) {
        reduce(rows, input_cols, lookup, survived_cols);
        ptr = &survived_cols;
    }
    auto& cols = *ptr; // avoid memory copy

    /**********************************
        * INTERPOLATE
        **********************************/

        // call recursively on odd-indexed rows
    std::vector<id_t> odd_rows;
    for (id_t i = 1; i < rows.size(); i += 2) {
        odd_rows.push_back(rows[i]);
    }
    smawk_impl(odd_rows, cols, lookup, argmins);

    // interpolate the even-indexed rows
    interpolate(rows, cols, lookup, argmins);
}

void smawk(const id_t n, const std::function<float(id_t, id_t)>& lookup, id_t* argmins)
{
    std::vector<id_t> rows(n);
    std::vector<id_t> cols(n);
    std::iota(std::begin(rows), std::end(rows), 0);
    std::iota(std::begin(cols), std::end(cols), 0);

    smawk_impl(rows, cols, lookup, argmins);
}

}