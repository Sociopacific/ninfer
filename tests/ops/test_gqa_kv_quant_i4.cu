// Bit-exactness of the INT4-G32 KV codec against a host oracle written straight
// from the numerical contract in include/ninfer/ops/gqa_attention.h.

#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kGroup  = ninfer::ops::kGqaKvQuantGroupI4;
constexpr int kGroups = 4096;
constexpr int kValues = kGroup * kGroups;

struct KvQuantGeometry {
    static constexpr int KVHeads = 4;
};

__global__ void encode_kernel(const __nv_bfloat16* source, std::uint8_t* codes,
                              const __half* scales) {
    const int group = blockIdx.x;
    const float s   = __half2float(scales[group]);
    const float inv = s == 0.0f ? 0.0f : 1.0f / s;
    for (int i = threadIdx.x; i < kGroup / 2; i += blockDim.x) {
        const float even = __bfloat162float(source[group * kGroup + 2 * i]);
        const float odd  = __bfloat162float(source[group * kGroup + 2 * i + 1]);
        codes[group * (kGroup / 2) + i] =
            ninfer::ops::gqa_kv_pack_i4x2(ninfer::ops::gqa_kv_quant_code_i4(even, inv),
                                          ninfer::ops::gqa_kv_quant_code_i4(odd, inv));
    }
}

__global__ void decode_kernel(const std::uint8_t* codes, const __half* scales,
                              std::int8_t* unpacked, __nv_bfloat16* dequantized) {
    const int group = blockIdx.x;
    const float s   = __half2float(scales[group]);
    for (int i = threadIdx.x; i < kGroup / 8; i += blockDim.x) {
        const std::uint8_t* src = &codes[group * (kGroup / 2) + i * 4];
        const int2 raw          = ninfer::ops::gqa_kv_unpack_i4x8_to_i8(src);
        ninfer::ops::store_vec(&unpacked[group * kGroup + i * 8], raw);
        const int4 values = ninfer::ops::gqa_kv_dequant_i4x8_from(src, s);
        ninfer::ops::store_vec(&dequantized[group * kGroup + i * 8], values);
    }
}

// The index helpers are pure arithmetic; evaluate them on device and compare with
// the plane geometry the layout planner builds.
__global__ void index_kernel(std::int64_t* out) {
    out[0] = ninfer::ops::gqa_kv_quant_code_index_i4<KvQuantGeometry>(3, 2, 130, 17);
    out[1] = ninfer::ops::gqa_kv_quant_scale_index_i4<KvQuantGeometry>(3, 2, 5, 17);
}

int failures = 0;

void fail(const char* what) {
    std::fprintf(stderr, "%s\n", what);
    ++failures;
}

float bf16_round(float value) { return __bfloat162float(__float2bfloat16(value)); }

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    const auto lower     = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) { return lower; }
    if (fraction > 0.5f) { return lower + 1; }
    return (lower & 1) == 0 ? lower : lower + 1;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 77;
    }

    // A spread that exercises every code, both signs, exact halfway ties, and a
    // group whose values are all zero (scale 0 must decode to zero, not NaN).
    std::vector<__nv_bfloat16> source(kValues);
    std::vector<float> logical(kValues);
    std::uint32_t state = 0x1234567u;
    for (int i = 0; i < kValues; ++i) {
        state              = state * 1664525u + 1013904223u;
        const int group    = i / kGroup;
        float value        = 0.0f;
        if (group % 8 != 7) {
            const float unit = static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
            // Halfway ties land on .5 exactly for a scale that is a power of two.
            value = (unit * 2.0f - 1.0f) * static_cast<float>(1 << (group % 5));
            if ((i % 17) == 0) { value = std::ldexp(std::round(unit * 15.0f) - 7.5f, group % 3); }
        }
        source[i]  = __float2bfloat16(value);
        logical[i] = __bfloat162float(source[i]);
    }

    // Host oracle: scale from the group absmax, FP16-rounded, then RNE codes.
    std::vector<__half> scales(kGroups);
    std::vector<std::uint8_t> expected_codes(kValues / 2);
    std::vector<std::int8_t> expected_unpacked(kValues);
    std::vector<float> expected_dequant(kValues);
    for (int group = 0; group < kGroups; ++group) {
        float absmax = 0.0f;
        for (int i = 0; i < kGroup; ++i) {
            absmax = std::max(absmax, std::fabs(logical[group * kGroup + i]));
        }
        const __half scale_bits = __float2half(absmax / 7.0f);
        const float stored      = __half2float(scale_bits);
        const float inverse     = stored == 0.0f ? 0.0f : 1.0f / stored;
        scales[group]           = scale_bits;
        for (int i = 0; i < kGroup; ++i) {
            std::int32_t code = 0;
            if (stored != 0.0f) {
                code = std::clamp(round_even_to_i32(logical[group * kGroup + i] * inverse), -7, 7);
            }
            expected_unpacked[group * kGroup + i] = static_cast<std::int8_t>(code);
            expected_dequant[group * kGroup + i] =
                bf16_round(static_cast<float>(code) * stored);
            if ((i & 1) == 0) {
                expected_codes[(group * kGroup + i) / 2] = static_cast<std::uint8_t>(code & 0xf);
            } else {
                expected_codes[(group * kGroup + i) / 2] |=
                    static_cast<std::uint8_t>((code & 0xf) << 4);
            }
        }
    }

    __nv_bfloat16* d_source     = nullptr;
    std::uint8_t* d_codes       = nullptr;
    __half* d_scales            = nullptr;
    std::int8_t* d_unpacked     = nullptr;
    __nv_bfloat16* d_dequant    = nullptr;
    std::int64_t* d_index       = nullptr;
    cudaMalloc(&d_source, source.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_codes, expected_codes.size());
    cudaMalloc(&d_scales, scales.size() * sizeof(__half));
    cudaMalloc(&d_unpacked, kValues);
    cudaMalloc(&d_dequant, kValues * sizeof(__nv_bfloat16));
    cudaMalloc(&d_index, 2 * sizeof(std::int64_t));
    cudaMemcpy(d_source, source.data(), source.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(__half), cudaMemcpyHostToDevice);

    encode_kernel<<<kGroups, 32>>>(d_source, d_codes, d_scales);
    decode_kernel<<<kGroups, 32>>>(d_codes, d_scales, d_unpacked, d_dequant);
    index_kernel<<<1, 1>>>(d_index);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<std::uint8_t> codes(expected_codes.size());
    std::vector<std::int8_t> unpacked(kValues);
    std::vector<__nv_bfloat16> dequant(kValues);
    std::int64_t index[2] = {0, 0};
    cudaMemcpy(codes.data(), d_codes, codes.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(unpacked.data(), d_unpacked, kValues, cudaMemcpyDeviceToHost);
    cudaMemcpy(dequant.data(), d_dequant, kValues * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(index, d_index, sizeof(index), cudaMemcpyDeviceToHost);

    if (codes != expected_codes) { fail("packed INT4 codes differ from the host oracle"); }
    if (unpacked != expected_unpacked) { fail("unpacked INT4 codes differ from the host oracle"); }
    for (int i = 0; i < kValues; ++i) {
        if (__bfloat162float(dequant[i]) != expected_dequant[i]) {
            fail("dequantized INT4 values differ from the host oracle");
            break;
        }
    }

    // Plane geometry: codes [128, 64, KVHeads, pages], scales [8, 64, KVHeads, pages].
    const std::int64_t expected_code_index = 128LL * 64 * (2 + 4LL * 3) + 128LL * 17 + 65;
    const std::int64_t expected_scale_index = 8LL * 64 * (2 + 4LL * 3) + 8LL * 17 + 5;
    if (index[0] != expected_code_index) { fail("INT4 code index does not match the plane"); }
    if (index[1] != expected_scale_index) { fail("INT4 scale index does not match the plane"); }

    cudaFree(d_source);
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_unpacked);
    cudaFree(d_dequant);
    cudaFree(d_index);

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
