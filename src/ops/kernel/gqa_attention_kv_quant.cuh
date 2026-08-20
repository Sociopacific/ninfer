#pragma once

// ninfer::ops - signed, per-token group-wise KV cache codec (shared device
// helpers) for both INT8-G64 and INT4-G32. Quantization (append) and
// dequantization (stage) are FUSED into the GQA attention kernels themselves
// (decode partial kernel, prefill fill/attention); this header only provides the
// index math, the vectorized dequant, and the scalar quantize helper they share.
// There is deliberately no standalone quant/dequant kernel: that would defeat the
// halved-bandwidth goal.
//
// INT4 stores two signed 4-bit codes per byte, the even dimension in the low
// nibble, in a plane of half the head extent. Codes are unpacked to int8 on the
// way into shared memory so the m16n8k32.s8 QK MMA is untouched by the narrower
// storage; V is expanded from nibbles straight to bf16, as INT8 already does.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

inline constexpr int kGqaKvQuantGroupI4  = 32;
inline constexpr int kGqaKvQuantGroupsI4 = kGqaKvQuantHeadDim / kGqaKvQuantGroupI4;
inline constexpr int kGqaKvQuantHeadBytesI4 = kGqaKvQuantHeadDim / 2;
// Symmetric like INT8's +-127: the -8 code is never emitted, so decode stays odd.
inline constexpr int kGqaKvQuantMaxI4 = 7;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

// Byte index of the pair that holds dimension `d`. `d` must be even for a run of
// codes to start on a byte boundary; every caller stages at least two at a time.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index_i4(int physical_page, int kv_head,
                                                                   int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadBytesI4, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index_i4(int physical_page, int kv_head,
                                                                    int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroupsI4, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/7). Same round-to-nearest-even and symmetric clamp as INT8.
__device__ __forceinline__ int gqa_kv_quant_code_i4(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0; }
    const int q = __float2int_rn(x * inv_scale);
    return max(-kGqaKvQuantMaxI4, min(kGqaKvQuantMaxI4, q));
}

// Pack two codes into one byte, the even dimension in the low nibble.
__device__ __forceinline__ std::uint8_t gqa_kv_pack_i4x2(int even, int odd) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(even) & 0xfu) |
                                     ((static_cast<unsigned>(odd) & 0xfu) << 4));
}

// Sign-extend four 4-bit lanes held in the low nibble of each byte. A lane whose
// sign bit is set gains 0xf0, which is an addition rather than a subtraction and
// so cannot borrow into the neighbouring lane.
__device__ __forceinline__ unsigned gqa_kv_sign_extend_i4x4(unsigned nibbles) {
    const unsigned sign = (nibbles >> 3) & 0x01010101u;
    return nibbles + sign * 0xf0u;
}

// Expand 8 packed codes (dimensions [d, d+8), d a multiple of 8) into 8 int8 in
// dimension order. One 32-bit load, seven ALU ops; the pointer may be global or
// shared. This is what keeps the QK MMA operating on the same int8 tile as before.
__device__ __forceinline__ int2 gqa_kv_unpack_i4x8_to_i8(const std::uint8_t* codes4) {
    const unsigned raw  = load_vec<unsigned>(codes4);
    const unsigned even = gqa_kv_sign_extend_i4x4(raw & 0x0f0f0f0fu);
    const unsigned odd  = gqa_kv_sign_extend_i4x4((raw >> 4) & 0x0f0f0f0fu);
    // Interleave back to d, d+1, ... : bytes (even0, odd0, even1, odd1), then (2,2,3,3).
    return make_int2(static_cast<int>(__byte_perm(even, odd, 0x5140u)),
                     static_cast<int>(__byte_perm(even, odd, 0x7362u)));
}

// Dequantize 8 packed codes into 8 bf16 packed as an int4, mirroring the INT8
// helper above so the two storages share the V staging path.
__device__ __forceinline__ int4 gqa_kv_dequant_i4x8_from(const std::uint8_t* codes4, float s) {
    const int2 codes     = gqa_kv_unpack_i4x8_to_i8(codes4);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&codes);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
