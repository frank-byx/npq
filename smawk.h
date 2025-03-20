/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * The code below is adapated from code from the FAISS repository licensed under the MIT license
 * found in the LICENSE file in the root directory of the FAISS repository.
 */

#pragma once

#include <functional>

#include "dataset.h"


namespace npq
{

/** SMAWK algorithm for an implicit n times n totally monotone matrix C.
 *
 * @param n         number of rows and columns
 * @param lookup    implicit matrix C, size (n, n)
 * @param argmins   argmin of each row of C
 */
void smawk(
    const id_t n,
    const std::function<float(id_t, id_t)>& lookup,
    id_t* argmins);

} // namespace npq