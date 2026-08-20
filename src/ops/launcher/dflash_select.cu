#include "ops/launcher/dflash_select.h"

#include "ops/kernel/dflash_select.cuh"
#include "ops/common/math.h"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kProjectBlock = 256;
constexpr int kTopkBlock    = 256;
constexpr int kWalkBlock    = 512;

} // namespace

void dflash_select_launch(const Tensor& logits, const Tensor& hidden, const Tensor& projection,
                          const Tensor& anchors, const Tensor& predecessor,
                          const Tensor& successor, std::int32_t valid_rows,
                          std::int32_t draft_tokens, std::int32_t rows, float* projected,
                          float* unary, std::int32_t* candidates, Tensor& drafts,
                          cudaStream_t stream) {
    const std::int32_t columns       = draft_tokens * rows;
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t features      = hidden.ne[0];
    const std::int32_t rank          = projection.ne[1];

    constexpr int kProjectWarps = kProjectBlock / 32;
    const dim3 project_grid(static_cast<unsigned>(columns),
                            static_cast<unsigned>(div_up(rank, kProjectWarps)));
    dflash_select_project_kernel<kProjectBlock><<<project_grid, kProjectBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(projection.data), features, rank, projected);
    CUDA_CHECK(cudaGetLastError());

    dflash_select_topk_kernel<kTopkBlock><<<static_cast<unsigned>(columns), kTopkBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), valid_rows, physical_rows, unary,
        candidates);
    CUDA_CHECK(cudaGetLastError());

    dflash_select_walk_kernel<kWalkBlock><<<static_cast<unsigned>(rows), kWalkBlock, 0, stream>>>(
        unary, candidates, projected, static_cast<const std::int32_t*>(anchors.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(successor.data), rank, draft_tokens,
        static_cast<std::int32_t*>(drafts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
