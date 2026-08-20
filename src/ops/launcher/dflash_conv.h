#pragma once

#include "core/tensor.h"
#include "ninfer/ops/dflash_conv.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void dflash_conv_launch(const Tensor& hidden, const Tensor& base, const Tensor& dynamic,
                        DFlashConvStage stage, bool accumulate, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops::detail
