// ninfer::ops - dflash_select wrapper: implements the public api, validates parameters, carves
// the transient scratch, and dispatches to the launcher. Host-compiled; never includes the
// kernel header.
#include "ninfer/ops/dflash_select.h"

#include "ops/launcher/dflash_select.h" // detail::dflash_select_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTop           = 16;
constexpr std::int32_t kMaximumRank   = 512;
constexpr std::int32_t kMaximumDrafts = 16;
constexpr std::int32_t kMaximumRows   = 8;

void require(const Tensor& tensor, DType dtype, const char* what) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("dflash_select: ") + what +
                                    " has the wrong dtype");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("dflash_select: ") + what + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("dflash_select: ") + what +
                                    " data must be non-null");
    }
}

} // namespace

std::size_t dflash_select_workspace_capacity_bytes(std::int32_t rank, std::int32_t max_columns) {
    if (rank < 1 || rank > kMaximumRank) {
        throw std::invalid_argument("dflash_select workspace: invalid rank");
    }
    if (max_columns < 1 || max_columns > kMaximumDrafts * kMaximumRows) {
        throw std::invalid_argument("dflash_select workspace: invalid column count");
    }
    // The FP32 rank projection, then a float and an int32 per candidate slot, all per column.
    return static_cast<std::size_t>(max_columns) *
           (static_cast<std::size_t>(rank) * sizeof(float) +
            kTop * (sizeof(float) + sizeof(std::int32_t)));
}

void dflash_select(const Tensor& logits, const Tensor& hidden, const Tensor& projection,
                   const Tensor& anchors, const Tensor& predecessor, const Tensor& successor,
                   std::int32_t valid_rows, std::int32_t draft_tokens, Tensor& drafts,
                   WorkspaceArena& workspace, cudaStream_t stream) {
    require(logits, DType::BF16, "logits");
    require(hidden, DType::BF16, "hidden");
    require(projection, DType::BF16, "projection");
    require(anchors, DType::I32, "anchors");
    require(predecessor, DType::BF16, "predecessor");
    require(successor, DType::BF16, "successor");
    require(drafts, DType::I32, "drafts");
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        projection.ne[2] != 1 || projection.ne[3] != 1 || predecessor.ne[2] != 1 ||
        predecessor.ne[3] != 1 || successor.ne[2] != 1 || successor.ne[3] != 1) {
        throw std::invalid_argument("dflash_select: matrix arguments must be two-dimensional");
    }
    const std::int32_t rows = anchors.ne[0];
    if (rows < 1 || rows > kMaximumRows || anchors.numel() != rows) {
        throw std::invalid_argument("dflash_select: anchors must be [B] with 1 <= B <= 8");
    }
    if (draft_tokens < 1 || draft_tokens > kMaximumDrafts) {
        throw std::invalid_argument("dflash_select: draft count must lie in [1,16]");
    }
    const std::int64_t columns = static_cast<std::int64_t>(draft_tokens) * rows;
    if (logits.ne[1] != columns || hidden.ne[1] != columns || drafts.numel() != columns) {
        throw std::invalid_argument("dflash_select: column extents must equal draft_tokens * B");
    }
    const std::int32_t physical_rows = logits.ne[0];
    if (valid_rows < kTop || valid_rows > physical_rows) {
        throw std::invalid_argument("dflash_select: valid rows must lie in [16, physical rows]");
    }
    if (projection.ne[0] != hidden.ne[0]) {
        throw std::invalid_argument("dflash_select: projection must share the hidden features");
    }
    const std::int32_t rank = projection.ne[1];
    if (rank < 1 || rank > kMaximumRank) {
        throw std::invalid_argument("dflash_select: rank must lie in [1,512]");
    }
    if (predecessor.ne[0] != rank || successor.ne[0] != rank) {
        throw std::invalid_argument("dflash_select: codebooks must share the projection rank");
    }
    if (predecessor.ne[1] < valid_rows || successor.ne[1] < valid_rows) {
        throw std::invalid_argument("dflash_select: codebooks must cover every valid row");
    }
    if (drafts.data == logits.data || drafts.data == hidden.data ||
        drafts.data == projection.data || drafts.data == anchors.data ||
        drafts.data == predecessor.data || drafts.data == successor.data) {
        throw std::invalid_argument("dflash_select: drafts must not alias an input");
    }

    auto scratch_scope      = workspace.scope();
    const std::size_t bytes = dflash_select_workspace_capacity_bytes(
        rank, static_cast<std::int32_t>(columns));
    const DeviceSpan scratch = workspace.alloc_bytes(bytes);
    auto* projected          = static_cast<float*>(scratch.data);
    auto* unary = projected + static_cast<std::size_t>(columns) * static_cast<std::size_t>(rank);
    auto* candidates =
        reinterpret_cast<std::int32_t*>(unary + static_cast<std::size_t>(columns) * kTop);

    detail::dflash_select_launch(logits, hidden, projection, anchors, predecessor, successor,
                                 valid_rows, draft_tokens, rows, projected, unary, candidates,
                                 drafts, stream);
}

} // namespace ninfer::ops
