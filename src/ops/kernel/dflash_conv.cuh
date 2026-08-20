#pragma once

// Implements: include/ninfer/ops/dflash_conv.h
// Match: contiguous BF16 [D,W,B] whose group size is a multiple of the pack
// width, so one 16-byte channel pack carries a single group index and reads one
// dynamic weight per tap; scalar indexing covers the remaining geometries. Taps
// accumulate in FP32 in increasing tap order and round once on store.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <int Block, int Taps>
__launch_bounds__(Block) __global__
    void dflash_conv_bf16x8_kernel(const Bf16x8Pack* hidden, const Bf16x8Pack* base,
                                   const __nv_bfloat16* dynamic, Bf16x8Pack* y,
                                   std::int32_t packs, std::int32_t packs_per_group,
                                   std::int32_t groups, std::int32_t width,
                                   std::int32_t dynamic_rows, bool accumulate) {
    const std::int32_t pack =
        static_cast<std::int32_t>(blockIdx.x) * Block + static_cast<std::int32_t>(threadIdx.x);
    if (pack >= packs) { return; }
    const std::int32_t position = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t column   = static_cast<std::int64_t>(blockIdx.z) * width + position;
    const std::int32_t group    = pack / packs_per_group;

    float sum[8];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) { sum[lane] = 0.0F; }
#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        if (position < tap) { continue; }
        const Bf16x8Pack values = load_vec<Bf16x8Pack>(hidden + (column - tap) * packs + pack);
        const Bf16x8Pack statics =
            load_vec<Bf16x8Pack>(base + static_cast<std::int64_t>(tap) * packs + pack);
        const float dynamic_weight = __bfloat162float(
            dynamic[column * dynamic_rows + static_cast<std::int64_t>(tap) * groups + group]);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            sum[2 * pair] += (__low2float(statics.pair[pair]) + dynamic_weight) *
                             __low2float(values.pair[pair]);
            sum[2 * pair + 1] += (__high2float(statics.pair[pair]) + dynamic_weight) *
                                 __high2float(values.pair[pair]);
        }
    }
    const std::int64_t destination = column * packs + pack;
    if (accumulate) {
        const Bf16x8Pack previous = load_vec<Bf16x8Pack>(y + destination);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            sum[2 * pair] += __low2float(previous.pair[pair]);
            sum[2 * pair + 1] += __high2float(previous.pair[pair]);
        }
    }
    Bf16x8Pack result;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        result.pair[pair] = __floats2bfloat162_rn(sum[2 * pair], sum[2 * pair + 1]);
    }
    store_vec(y + destination, result);
}

__global__ void dflash_conv_kernel(const __nv_bfloat16* hidden, const __nv_bfloat16* base,
                                   const __nv_bfloat16* dynamic, __nv_bfloat16* y,
                                   std::int32_t channels, std::int32_t channels_per_group,
                                   std::int32_t groups, std::int32_t width,
                                   std::int32_t dynamic_rows, std::int32_t taps,
                                   std::int64_t total, bool accumulate) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t index = start; index < total; index += stride) {
        const std::int32_t channel  = static_cast<std::int32_t>(index % channels);
        const std::int64_t column   = index / channels;
        const std::int32_t position = static_cast<std::int32_t>(column % width);
        const std::int32_t group    = channel / channels_per_group;
        float sum                   = 0.0F;
        for (std::int32_t tap = 0; tap < taps; ++tap) {
            if (position < tap) { break; }
            const float value   = __bfloat162float(hidden[(column - tap) * channels + channel]);
            const float statics =
                __bfloat162float(base[static_cast<std::int64_t>(tap) * channels + channel]);
            const float dynamic_weight = __bfloat162float(
                dynamic[column * dynamic_rows + static_cast<std::int64_t>(tap) * groups + group]);
            sum += (statics + dynamic_weight) * value;
        }
        if (accumulate) { sum += __bfloat162float(y[index]); }
        y[index] = __float2bfloat16_rn(sum);
    }
}

} // namespace ninfer::ops
