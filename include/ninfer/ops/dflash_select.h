#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Caller-owned transient capacity for a dflash_select() call with the given codebook rank over
// at most max_columns proposal columns. Invalid arguments throw.
[[nodiscard]] std::size_t dflash_select_workspace_capacity_bytes(std::int32_t rank,
                                                                 std::int32_t max_columns);

/**
 * Op: dflash_select
 *
 * Chains a block of draft tokens through the DFlash2 candidate selector. Columns are grouped by
 * proposal row: column t = i + K*b holds draft position i of row b, with K = draft_tokens and
 * B = anchors' extent. First every column is projected into the codebook rank space,
 *
 *   projected[r,t] = sum_i float(hidden[i,t]) * float(projection[i,r]),
 *
 * then within a row the walk is sequential; prev starts at anchors[b], and for i = 0..K-1 in
 * order with t = i + K*b:
 *
 *   C = the 16 rows with the largest logits[v,t] over 0 <= v < valid_rows
 *       (equal values prefer the lower row index);
 *   score[c] = float(logits[C[c],t])
 *            + sum_r projected[r,t] * float(predecessor[r,prev]) * float(successor[r,C[c]]);
 *   drafts[t] = C[argmax_c score[c]], prev = drafts[t],
 *
 * where equal scores prefer the candidate that sorts earlier in C (larger logit, then lower
 * token id). Projection and scores accumulate in FP32 from BF16-represented operands.
 *
 * Logical shapes:
 *   logits is contiguous BF16 [physical_rows, K*B] with 16 <= valid_rows <= physical_rows;
 *   physical rows [valid_rows, physical_rows) do not participate. hidden is contiguous BF16
 *   [features, K*B] and projection is contiguous BF16 [features, rank] with 1 <= rank <= 512.
 *   predecessor and successor are contiguous BF16 [rank, vocabulary] with vocabulary >=
 *   valid_rows; anchors is device I32 [B] whose values must lie in [0, vocabulary). drafts is
 *   contiguous I32 [K*B] with 1 <= K <= 16 and 1 <= B <= 8. drafts must not overlap any input.
 *   The workspace is transient scratch; no other state changes.
 */
void dflash_select(const Tensor& logits, const Tensor& hidden, const Tensor& projection,
                   const Tensor& anchors, const Tensor& predecessor, const Tensor& successor,
                   std::int32_t valid_rows, std::int32_t draft_tokens, Tensor& drafts,
                   WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
