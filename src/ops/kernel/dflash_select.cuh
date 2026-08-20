#pragma once

// Implements: include/ninfer/ops/dflash_select.h
// Match: the projection assigns one warp per (rank, column) pair; both operands stream
// contiguously along the feature axis. The top-k runs one block per column keeping a register
// top-16 per thread (a predicated bubble insert avoids dynamic register indexing), then peels
// the block maximum sixteen times; the winner is always slot zero of its owner, so the removal
// shift is statically unrolled too. The walk runs one block per proposal row: the
// predecessor-weighted projection lands in shared memory once per position and one warp per
// candidate reduces the rank axis.

#include <cuda_bf16.h>

#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kDFlashSelectTop      = 16;
inline constexpr int kDFlashSelectMaxRank  = 512;

namespace detail_dflash_select {

__device__ __forceinline__ bool candidate_before(float value, std::int32_t id, float other_value,
                                                 std::int32_t other_id) {
    return value > other_value || (value == other_value && id < other_id);
}

} // namespace detail_dflash_select

template <int Block>
__launch_bounds__(Block) __global__
    void dflash_select_project_kernel(const __nv_bfloat16* hidden,
                                      const __nv_bfloat16* projection, std::int32_t features,
                                      std::int32_t rank, float* projected) {
    constexpr int kWarps    = Block / 32;
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t r    = static_cast<std::int32_t>(blockIdx.y) * kWarps +
                           (static_cast<std::int32_t>(threadIdx.x) >> 5);
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x) & 31;
    if (r >= rank) { return; }
    const __nv_bfloat16* column = hidden + static_cast<std::int64_t>(t) * features;
    const __nv_bfloat16* row    = projection + static_cast<std::int64_t>(r) * features;
    float sum                   = 0.0F;
    for (std::int32_t i = lane; i < features; i += 32) {
        sum += __bfloat162float(column[i]) * __bfloat162float(row[i]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) { projected[static_cast<std::int64_t>(t) * rank + r] = sum; }
}

template <int Block>
__launch_bounds__(Block) __global__
    void dflash_select_topk_kernel(const __nv_bfloat16* logits, std::int32_t valid_rows,
                                   std::int32_t physical_rows, float* unary,
                                   std::int32_t* candidates) {
    using detail_dflash_select::candidate_before;
    constexpr int kTop           = kDFlashSelectTop;
    constexpr std::int32_t kNone = 0x7fffffff;
    const std::int64_t column    = static_cast<std::int64_t>(blockIdx.x);
    const __nv_bfloat16* source  = logits + column * physical_rows;

    float value[kTop];
    std::int32_t id[kTop];
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
        value[j] = -CUDART_INF_F;
        id[j]    = kNone;
    }
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += Block) {
        const float x = __bfloat162float(source[v]);
        if (!candidate_before(x, v, value[kTop - 1], id[kTop - 1])) { continue; }
        float carry_value      = x;
        std::int32_t carry_id  = v;
#pragma unroll
        for (int j = 0; j < kTop; ++j) {
            if (candidate_before(carry_value, carry_id, value[j], id[j])) {
                const float evicted_value     = value[j];
                const std::int32_t evicted_id = id[j];
                value[j]                      = carry_value;
                id[j]                         = carry_id;
                carry_value                   = evicted_value;
                carry_id                      = evicted_id;
            }
        }
    }

    constexpr int kWarps = Block / 32;
    __shared__ float warp_values[kWarps];
    __shared__ std::int32_t warp_ids[kWarps];
    __shared__ std::int32_t winner_id;
    for (int round = 0; round < kTop; ++round) {
        float best_value     = value[0];
        std::int32_t best_id = id[0];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value     = __shfl_down_sync(0xffffffffU, best_value, offset);
            const std::int32_t other_id = __shfl_down_sync(0xffffffffU, best_id, offset);
            if (candidate_before(other_value, other_id, best_value, best_id)) {
                best_value = other_value;
                best_id    = other_id;
            }
        }
        if ((threadIdx.x & 31U) == 0U) {
            warp_values[threadIdx.x >> 5] = best_value;
            warp_ids[threadIdx.x >> 5]    = best_id;
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            float block_value     = warp_values[0];
            std::int32_t block_id = warp_ids[0];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) {
                if (candidate_before(warp_values[w], warp_ids[w], block_value, block_id)) {
                    block_value = warp_values[w];
                    block_id    = warp_ids[w];
                }
            }
            winner_id                          = block_id;
            unary[column * kTop + round]       = block_value;
            candidates[column * kTop + round]  = block_id;
        }
        __syncthreads();
        if (id[0] == winner_id) {
#pragma unroll
            for (int j = 0; j + 1 < kTop; ++j) {
                value[j] = value[j + 1];
                id[j]    = id[j + 1];
            }
            value[kTop - 1] = -CUDART_INF_F;
            id[kTop - 1]    = kNone;
        }
        __syncthreads();
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void dflash_select_walk_kernel(const float* unary, const std::int32_t* candidates,
                                   const float* projected, const std::int32_t* anchors,
                                   const __nv_bfloat16* predecessor,
                                   const __nv_bfloat16* successor, std::int32_t rank,
                                   std::int32_t draft_tokens, std::int32_t* drafts) {
    constexpr int kTop      = kDFlashSelectTop;
    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t warp = static_cast<std::int32_t>(threadIdx.x) >> 5;
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x) & 31;

    __shared__ float weighted[kDFlashSelectMaxRank];
    __shared__ float scores[kTop];
    __shared__ std::int32_t previous;
    if (threadIdx.x == 0U) { previous = anchors[row]; }
    __syncthreads();

    for (std::int32_t i = 0; i < draft_tokens; ++i) {
        const std::int64_t column =
            static_cast<std::int64_t>(i) + static_cast<std::int64_t>(draft_tokens) * row;
        for (std::int32_t r = static_cast<std::int32_t>(threadIdx.x); r < rank; r += Block) {
            weighted[r] =
                projected[column * rank + r] *
                __bfloat162float(predecessor[static_cast<std::int64_t>(previous) * rank + r]);
        }
        __syncthreads();
        if (warp < kTop) {
            const std::int32_t candidate = candidates[column * kTop + warp];
            float sum                    = 0.0F;
            for (std::int32_t r = lane; r < rank; r += 32) {
                sum += weighted[r] *
                       __bfloat162float(successor[static_cast<std::int64_t>(candidate) * rank + r]);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            }
            if (lane == 0) { scores[warp] = unary[column * kTop + warp] + sum; }
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            std::int32_t best = 0;
            float best_value  = scores[0];
#pragma unroll
            for (int c = 1; c < kTop; ++c) {
                if (scores[c] > best_value) {
                    best_value = scores[c];
                    best       = c;
                }
            }
            const std::int32_t token = candidates[column * kTop + best];
            drafts[column]           = token;
            previous                 = token;
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops
