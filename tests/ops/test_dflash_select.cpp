#include "ninfer/ops/dflash_select.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kTop = 16;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

struct Shape {
    std::int32_t draft_tokens;
    std::int32_t rows;
    std::int32_t valid_rows;
    std::int32_t physical_rows;
    std::int32_t features;
    std::int32_t rank;
    std::int32_t vocabulary;
};

// The oracle mirrors the Op contract in FP64: the rank projection, a value-sorted top-16 whose
// ties prefer the lower token id, then the sequential codebook walk whose argmax keeps the first
// maximum in that candidate order.
std::vector<std::int32_t> dflash_select_oracle(const Shape& shape,
                                               const std::vector<float>& logits,
                                               const std::vector<float>& hidden,
                                               const std::vector<float>& projection,
                                               const std::vector<std::int32_t>& anchors,
                                               const std::vector<float>& predecessor,
                                               const std::vector<float>& successor) {
    const std::size_t columns = static_cast<std::size_t>(shape.draft_tokens) *
                                static_cast<std::size_t>(shape.rows);
    std::vector<double> projected(columns * static_cast<std::size_t>(shape.rank));
    for (std::size_t t = 0; t < columns; ++t) {
        for (std::int32_t r = 0; r < shape.rank; ++r) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < shape.features; ++i) {
                sum += static_cast<double>(
                           hidden[t * static_cast<std::size_t>(shape.features) +
                                  static_cast<std::size_t>(i)]) *
                       static_cast<double>(
                           projection[static_cast<std::size_t>(r) *
                                          static_cast<std::size_t>(shape.features) +
                                      static_cast<std::size_t>(i)]);
            }
            projected[t * static_cast<std::size_t>(shape.rank) + static_cast<std::size_t>(r)] =
                sum;
        }
    }

    std::vector<std::int32_t> drafts(columns);
    for (std::int32_t row = 0; row < shape.rows; ++row) {
        std::int32_t previous = anchors[static_cast<std::size_t>(row)];
        for (std::int32_t i = 0; i < shape.draft_tokens; ++i) {
            const std::size_t column = static_cast<std::size_t>(i) +
                                       static_cast<std::size_t>(shape.draft_tokens) *
                                           static_cast<std::size_t>(row);
            const float* column_logits =
                logits.data() + column * static_cast<std::size_t>(shape.physical_rows);
            std::vector<std::int32_t> order(static_cast<std::size_t>(shape.valid_rows));
            for (std::int32_t v = 0; v < shape.valid_rows; ++v) {
                order[static_cast<std::size_t>(v)] = v;
            }
            std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                              [&](std::int32_t a, std::int32_t b) {
                                  const float va = column_logits[a];
                                  const float vb = column_logits[b];
                                  if (va != vb) { return va > vb; }
                                  return a < b;
                              });
            double best_score       = 0.0;
            std::int32_t best_token = -1;
            for (std::int32_t c = 0; c < kTop; ++c) {
                const std::int32_t token = order[static_cast<std::size_t>(c)];
                double score             = static_cast<double>(column_logits[token]);
                for (std::int32_t r = 0; r < shape.rank; ++r) {
                    const double h = projected[column * static_cast<std::size_t>(shape.rank) +
                                               static_cast<std::size_t>(r)];
                    const double p = static_cast<double>(
                        predecessor[static_cast<std::size_t>(previous) *
                                        static_cast<std::size_t>(shape.rank) +
                                    static_cast<std::size_t>(r)]);
                    const double s = static_cast<double>(
                        successor[static_cast<std::size_t>(token) *
                                      static_cast<std::size_t>(shape.rank) +
                                  static_cast<std::size_t>(r)]);
                    score += h * p * s;
                }
                if (c == 0 || score > best_score) {
                    best_score = score;
                    best_token = token;
                }
            }
            drafts[column] = best_token;
            previous       = best_token;
        }
    }
    return drafts;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed) {
    const std::size_t columns = static_cast<std::size_t>(shape.draft_tokens) *
                                static_cast<std::size_t>(shape.rows);
    const std::size_t logit_count  = columns * static_cast<std::size_t>(shape.physical_rows);
    const std::size_t hidden_count = columns * static_cast<std::size_t>(shape.features);
    const std::size_t projection_count =
        static_cast<std::size_t>(shape.features) * static_cast<std::size_t>(shape.rank);
    const std::size_t codebook_count =
        static_cast<std::size_t>(shape.vocabulary) * static_cast<std::size_t>(shape.rank);

    std::vector<float> logits(logit_count), hidden(hidden_count), projection(projection_count),
        predecessor(codebook_count), successor(codebook_count);
    fill_uniform(logits, seed, -8.0f, 8.0f);
    fill_uniform(hidden, seed + 1, -1.0f, 1.0f);
    fill_uniform(projection, seed + 2, -0.25f, 0.25f);
    fill_uniform(predecessor, seed + 3, -0.5f, 0.5f);
    fill_uniform(successor, seed + 4, -0.5f, 0.5f);
    round_to_bf16(logits);
    round_to_bf16(hidden);
    round_to_bf16(projection);
    round_to_bf16(predecessor);
    round_to_bf16(successor);
    // Physical rows beyond valid_rows carry poison the top-k must never touch.
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::int32_t v = shape.valid_rows; v < shape.physical_rows; ++v) {
            logits[column * static_cast<std::size_t>(shape.physical_rows) +
                   static_cast<std::size_t>(v)] = 1024.0f;
        }
    }
    std::vector<std::int32_t> anchors(static_cast<std::size_t>(shape.rows));
    for (std::int32_t row = 0; row < shape.rows; ++row) {
        anchors[static_cast<std::size_t>(row)] =
            static_cast<std::int32_t>((seed + 7u * static_cast<std::uint32_t>(row)) %
                                      static_cast<std::uint32_t>(shape.vocabulary));
    }

    const auto expected =
        dflash_select_oracle(shape, logits, hidden, projection, anchors, predecessor, successor);

    const auto logit_bits       = encode_bf16(logits);
    const auto hidden_bits      = encode_bf16(hidden);
    const auto projection_bits  = encode_bf16(projection);
    const auto predecessor_bits = encode_bf16(predecessor);
    const auto successor_bits   = encode_bf16(successor);

    GuardedDeviceBuffer device_logits(logit_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_projection(projection_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_predecessor(predecessor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_successor(successor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_drafts(columns * sizeof(std::int32_t));
    device_logits.copy_from_host(logit_bits.data(), device_logits.bytes());
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_projection.copy_from_host(projection_bits.data(), device_projection.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_predecessor.copy_from_host(predecessor_bits.data(), device_predecessor.bytes());
    device_successor.copy_from_host(successor_bits.data(), device_successor.bytes());

    Tensor logits_tensor(device_logits.data(), DType::BF16,
                         {shape.physical_rows, static_cast<std::int32_t>(columns)});
    Tensor hidden_tensor(device_hidden.data(), DType::BF16,
                         {shape.features, static_cast<std::int32_t>(columns)});
    Tensor projection_tensor(device_projection.data(), DType::BF16,
                             {shape.features, shape.rank});
    Tensor anchors_tensor(device_anchors.data(), DType::I32, {shape.rows});
    Tensor predecessor_tensor(device_predecessor.data(), DType::BF16,
                              {shape.rank, shape.vocabulary});
    Tensor successor_tensor(device_successor.data(), DType::BF16, {shape.rank, shape.vocabulary});
    Tensor drafts_tensor(device_drafts.data(), DType::I32, {static_cast<std::int32_t>(columns)});

    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::dflash_select_workspace_capacity_bytes(shape.rank,
                                                         static_cast<std::int32_t>(columns))));
    ops::dflash_select(logits_tensor, hidden_tensor, projection_tensor, anchors_tensor,
                       predecessor_tensor, successor_tensor, shape.valid_rows, shape.draft_tokens,
                       drafts_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::int32_t>(device_drafts.data(), columns),
                                expected);
    failures += verify_exact("dflash_select logits unchanged",
                             from_device<std::uint16_t>(device_logits.data(), logit_bits.size()),
                             logit_bits);
    failures += device_logits.verify_guards("dflash_select logits");
    failures += device_hidden.verify_guards("dflash_select hidden");
    failures += device_projection.verify_guards("dflash_select projection");
    failures += device_anchors.verify_guards("dflash_select anchors");
    failures += device_predecessor.verify_guards("dflash_select predecessor");
    failures += device_successor.verify_guards("dflash_select successor");
    failures += device_drafts.verify_guards("dflash_select drafts");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // The registered proposal geometry: seven drafts, one lane, rank 256. Poisoned tail rows
    // check the valid-row clamp.
    failures += run_case("dflash_select [7,1] r256", {7, 1, 1000, 1024, 512, 256, 1024}, 101u);
    // Multiple rows walk independent anchors.
    failures += run_case("dflash_select [7,4] r256", {7, 4, 2048, 2048, 256, 256, 2048}, 201u);
    // A single position degenerates the walk to one selection per row; codebooks wider than the
    // head and a rank off the warp multiple are legal.
    failures += run_case("dflash_select [1,8] r96", {1, 8, 500, 512, 128, 96, 600}, 301u);
    // The smallest legal head: every valid row is a candidate.
    failures += run_case("dflash_select [3,2] valid=16", {3, 2, 16, 32, 64, 32, 40}, 401u);
    std::cout << (failures ? "FAIL" : "OK") << " dflash_select\n";
    return failures ? 1 : 0;
}
