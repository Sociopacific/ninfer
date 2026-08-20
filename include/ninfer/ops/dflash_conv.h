#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

/** Selects which half of the convolution parameters a call consumes. */
enum class DFlashConvStage : std::int32_t {
    Prepare = 0,
    Finish  = 1,
};

/**
 * Applies the grouped dynamic causal convolution of a DFlash2 draft layer along the block axis.
 *
 * With D channels partitioned into G groups of D/G adjacent channels, T taps, W positions per
 * block, B blocks and stage index s:
 *
 *   ideal[d,w,b] = sum over 0 <= t < T, in increasing t, of
 *       (base[d,t,s] + dynamic[s*T*G + t*G + d/(D/G), w, b]) * hidden[d,w-t,b],
 *
 * where taps reaching before the start of a block (w - t < 0) contribute nothing. `accumulate`
 * selects the write: false stores `ideal`, true stores `y[d,w,b] + ideal[d,w,b]` read from the
 * incoming y, which is how a layer folds its residual into the convolution.
 *
 * All tensors are contiguous BF16: `hidden` and `y` are [D,W,B], `base` is [D,T,2] and `dynamic`
 * is [2*T*G,W,B]; G is recovered from `dynamic` and must divide D. The oracle evaluates `ideal`
 * in FP64 from the represented inputs, summing taps in increasing t, and adds the promoted
 * incoming y when `accumulate` is set. The BF16 result is promoted and compared directly with
 * that; output storage rounding belongs to the Op's numerical criterion, not the oracle. Private
 * kernel arithmetic is implementation-defined. `hidden` must not overlap `y`, and neither `base`
 * nor `dynamic` may overlap either. There is no workspace or other state side effect.
 */
void dflash_conv(const Tensor& hidden, const Tensor& base, const Tensor& dynamic,
                 DFlashConvStage stage, bool accumulate, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
