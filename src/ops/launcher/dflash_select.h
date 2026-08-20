#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void dflash_select_launch(const Tensor& logits, const Tensor& hidden, const Tensor& projection,
                          const Tensor& anchors, const Tensor& predecessor,
                          const Tensor& successor, std::int32_t valid_rows,
                          std::int32_t draft_tokens, std::int32_t rows, float* projected,
                          float* unary, std::int32_t* candidates, Tensor& drafts,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail
