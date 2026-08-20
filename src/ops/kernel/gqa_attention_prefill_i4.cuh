#pragma once

// INT4-G32 GQA prompt kernel for the registered Qwen3.6 head geometries. It is the
// INT8 kernel with the storage narrowed: K nibbles are expanded to int8 in shared memory so the
// m16n8k32.s8 QK Tensor Core path is untouched, V is dequantized straight from nibbles to packed
// FP16 while producer warps execute QK, and Q is quantized at group 32 so its scale index lines up
// with K's. Sixteen warps split each 16-row FP16 PV output across four 64-dimension slices.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillI4Warps      = 16;
inline constexpr int kGqaPrefillI4Threads    = kGqaPrefillI4Warps * 32;
inline constexpr int kGqaPrefillI4Br         = 64;
inline constexpr int kGqaPrefillI4Bc         = 64;
inline constexpr int kGqaPrefillI4Groups     = kGqaPrefillHeadDim / kGqaKvQuantGroupI4;
inline constexpr int kGqaPrefillI4DB16       = kGqaPrefillHeadDim / 2;
inline constexpr int kGqaPrefillI4RowTiles   = kGqaPrefillI4Br / 16;
inline constexpr int kGqaPrefillI4DConsumers = kGqaPrefillI4Warps / kGqaPrefillI4RowTiles;

inline constexpr int kGqaPrefillI4QBytes = kGqaPrefillI4Br * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI4QScaleBytes =
    kGqaPrefillI4Br * kGqaPrefillI4Groups * static_cast<int>(sizeof(float));
// The QK MMA reads a full-width int8 tile; the nibbles land in a separate staging plane per
// side, and the two of them together are exactly the INT8 kernel's V code plane.
inline constexpr int kGqaPrefillI4KBytes    = kGqaPrefillI4Bc * kGqaPrefillHeadDim;
inline constexpr int kGqaPrefillI4CodeBytes = kGqaPrefillI4Bc * kGqaKvQuantHeadBytesI4;
inline constexpr int kGqaPrefillI4VStageBytes =
    kGqaPrefillI4Bc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4PBytes =
    kGqaPrefillI4Br * kGqaPrefillI4Bc * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4ScaleBytes =
    2 * kGqaPrefillI4Bc * kGqaPrefillI4Groups * static_cast<int>(sizeof(__half));
inline constexpr int kGqaPrefillI4StatsBytes =
    2 * kGqaPrefillI4Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillI4SmemBytes =
    kGqaPrefillI4QBytes + kGqaPrefillI4QScaleBytes + kGqaPrefillI4KBytes +
    2 * kGqaPrefillI4CodeBytes + kGqaPrefillI4VStageBytes + kGqaPrefillI4PBytes +
    kGqaPrefillI4ScaleBytes + kGqaPrefillI4StatsBytes;

static_assert(kGqaPrefillI4Groups == 8);
static_assert(kGqaPrefillI4DConsumers == 4);
// 2 KiB above INT8 and still under the 99 KiB opt-in ceiling, so occupancy is unchanged.
static_assert(kGqaPrefillI4SmemBytes == 94720);

// Dequantize 8 packed codes (dims [d, d+8), inside one 32-group) to packed FP16, mirroring the
// INT8 helper so the V staging path differs only in how the codes are read.
__device__ __forceinline__ int4 gqa_prefill_i4_dequant_f16x8(const std::uint8_t* codes4,
                                                             __half scale) {
    const int2 raw       = gqa_kv_unpack_i4x8_to_i8(codes4);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Eight independent quantization units per CTA; one warp owns one
// (token, kv_head, 32-d group), one dimension per lane. Odd lanes hand their code to the even
// neighbour, which writes the packed byte.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_i4_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                              std::uint8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaPrefillI4Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kGqaPrefillI4Groups;
    const int tmp                   = unit / kGqaPrefillI4Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    const int d0                    = group * kGqaKvQuantGroupI4 + lane;

    const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
    const float k0          = __bfloat162float(k[src0]);
    const float v0          = __bfloat162float(v[src0]);

    const float k_abs = warp_max(fabsf(k0), FullMask);
    const float v_abs = warp_max(fabsf(v0), FullMask);
    constexpr float Qmax = static_cast<float>(kGqaKvQuantMaxI4);

    const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / Qmax : 0.0f);
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / Qmax : 0.0f);
    const float ks   = __half2float(ksh);
    const float vs   = __half2float(vsh);
    const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    page             = __shfl_sync(FullMask, page, 0);

    const int k_code = gqa_kv_quant_code_i4(k0, kinv);
    const int v_code = gqa_kv_quant_code_i4(v0, vinv);
    const int k_odd  = __shfl_down_sync(FullMask, k_code, 1);
    const int v_odd  = __shfl_down_sync(FullMask, v_code, 1);
    if ((lane & 1) == 0) {
        const std::int64_t code_off =
            gqa_kv_quant_code_index_i4<Geometry>(page, kv_head, d0, page_off);
        cache_k[code_off] = gqa_kv_pack_i4x2(k_code, k_odd);
        cache_v[code_off] = gqa_kv_pack_i4x2(v_code, v_odd);
    }
    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index_i4<Geometry>(page, kv_head, group, page_off);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

// Large appends are scheduled in absolute eight-token tiles. Eight divides P=64, so each CTA is
// page-local while an unknown base offset costs at most one empty tail CTA in the launch envelope.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_i4_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int group             = static_cast<int>(blockIdx.z);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;

    const int token  = token_begin + warp;
    const bool valid = token < token_end;
    const int d0     = group * kGqaKvQuantGroupI4 + lane;
    float k0 = 0.0f, v0 = 0.0f;
    if (valid) {
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
        k0                      = __bfloat162float(k[src0]);
        v0                      = __bfloat162float(v[src0]);
    }
    const float k_abs    = warp_max(fabsf(k0), FullMask);
    const float v_abs    = warp_max(fabsf(v0), FullMask);
    constexpr float Qmax = static_cast<float>(kGqaKvQuantMaxI4);
    const __half ksh     = __float2half_rn(k_abs > 0.0f ? k_abs / Qmax : 0.0f);
    const __half vsh     = __float2half_rn(v_abs > 0.0f ? v_abs / Qmax : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;
    physical_page     = __shfl_sync(FullMask, physical_page, 0);
    if (!valid) { return; }

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const std::int64_t code_base =
        paged_kv_page_head_offset<kGqaKvQuantHeadBytesI4, Geometry::KVHeads>(physical_page,
                                                                             kv_head) +
        static_cast<std::int64_t>(page_off) * kGqaKvQuantHeadBytesI4 +
        group * (kGqaKvQuantGroupI4 / 2);
    const int k_code = gqa_kv_quant_code_i4(k0, kinv);
    const int v_code = gqa_kv_quant_code_i4(v0, vinv);
    const int k_odd  = __shfl_down_sync(FullMask, k_code, 1);
    const int v_odd  = __shfl_down_sync(FullMask, v_code, 1);
    if ((lane & 1) == 0) {
        cache_k[code_base + (lane >> 1)] = gqa_kv_pack_i4x2(k_code, k_odd);
        cache_v[code_base + (lane >> 1)] = gqa_kv_pack_i4x2(v_code, v_odd);
    }
    if (lane == 0) {
        const std::int64_t scale_offset =
            paged_kv_page_head_offset<kGqaKvQuantGroupsI4, Geometry::KVHeads>(physical_page,
                                                                              kv_head) +
            static_cast<std::int64_t>(page_off) * kGqaKvQuantGroupsI4 + group;
        scale_k[scale_offset] = ksh;
        scale_v[scale_offset] = vsh;
    }
}

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(120) void gqa_attention_prefill_i4_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillI4Br;
    constexpr int Bc            = kGqaPrefillI4Bc;
    constexpr int DB16          = kGqaPrefillI4DB16;
    constexpr int Groups        = kGqaPrefillI4Groups;
    constexpr int GroupKc       = kGqaKvQuantGroupI4 / 32;
    constexpr int DPacked       = kGqaKvQuantHeadBytesI4;
    // Producer warps keep this many Q group scales in registers; the rest reload per key tile.
    // Two is the spill-free point at 120 registers, INT8's finer group count notwithstanding.
    constexpr int QScaleCached = 2;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillI4DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kGqaPrefillI4RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillI4Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 1);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kGqaPrefillI4QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kGqaPrefillI4QScaleBytes);
    std::uint8_t* k_i4 = reinterpret_cast<std::uint8_t*>(k_i8 + kGqaPrefillI4KBytes);
    std::uint8_t* v_i4 = k_i4 + kGqaPrefillI4CodeBytes;
    __half* v_f16      = reinterpret_cast<__half*>(v_i4 + kGqaPrefillI4CodeBytes);
    __half* p_s       = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                  kGqaPrefillI4VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kGqaPrefillI4PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaPrefillI4Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One warp owns one (row, 64-d group) at a time.
    for (int unit = warp; unit < Br * Groups; unit += kGqaPrefillI4Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroupI4 + lane;
        float x0      = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row)]);
        }
        const float absmax = warp_max(fabsf(x0), FullMask);
        const float qs     = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv    = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_prefill_i8_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillI4Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_scale_index_i4<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<16>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<16>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        // 16 packed bytes cover 32 dimensions. The staging planes stay linear: the QK swizzle is
        // applied by the unpack pass, and V is read straight out at dequant time.
        for (int chunk = tid; chunk < Bc * (DPacked / 16); chunk += kGqaPrefillI4Threads) {
            const int key_l  = chunk / (DPacked / 16);
            const int dc     = chunk - key_l * (DPacked / 16);
            const int d      = dc * 32;
            const int key    = tile_k0 + key_l;
            std::uint8_t* kd = &k_i4[key_l * DPacked + (d >> 1)];
            std::uint8_t* vd = &v_i4[key_l * DPacked + (d >> 1)];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_kv_quant_code_index_i4<Geometry>(physical_page, kv_head, d, key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                cp_async<16, Cache::cg>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keeping exactly two group scales live is the spill-free 120-register point on SM120.
    // Groups 2/3 reload per key tile; retaining all four creates an 8-byte stack frame.
    float q_scale_r0[QScaleCached];
    float q_scale_r1[QScaleCached];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < QScaleCached; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        // Expand the staged nibbles into the int8 tile the QK MMA reads. Doing it here frees the
        // packed plane for the prefetch issued further down in this same iteration.
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kGqaPrefillI4Threads) {
            const int key_l         = chunk / (D / 16);
            const int dc            = chunk - key_l * (D / 16);
            const std::uint8_t* src = &k_i4[key_l * DPacked + dc * 8];
            const int2 lo           = gqa_kv_unpack_i4x8_to_i8(src);
            const int2 hi           = gqa_kv_unpack_i4x8_to_i8(src + 4);
            store_vec(&k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, dc * 8)) * 2],
                      make_int4(lo.x, lo.y, hi.x, hi.y));
        }
        __syncthreads();
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < QScaleCached) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 gqa_prefill_swz(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + gqa_prefill_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0          = __shfl_sync(FullMask, ks0, lid);
                    ks1          = __shfl_sync(FullMask, ks1, lid);
                    score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + gqa_prefill_i8_p_swz(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + gqa_prefill_i8_p_swz(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + gqa_prefill_i8_p_swz(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + gqa_prefill_i8_p_swz(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + gqa_prefill_swz(key_l, d)];
                if (key <= max_query_abs) {
                    const int grp = d >> 5;
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & 3) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                    vs = __shfl_sync(FullMask, vs, grp * 4);
                    store_vec(dst,
                              gqa_prefill_i4_dequant_f16x8(&v_i4[key_l * DPacked + (d >> 1)], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        const int row_tile = warp % kGqaPrefillI4RowTiles;
        const int d_slice  = warp / kGqaPrefillI4RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       gqa_prefill_i8_p_swz(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillI4RowTiles;
    const int d_slice  = warp / kGqaPrefillI4RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kGqaPrefillI4Threads);
}

} // namespace ninfer::ops
