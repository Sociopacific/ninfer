#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ResidentPrefixIdentity {
public:
    template<class Archive> void archive(Archive& ar) {
        ar(token_types_); for (auto& positions : positions_) ar(positions);
        std::uint64_t count = vision_items_.size(); ar(count); if (count > 1048576) throw std::runtime_error("invalid checkpoint media count"); vision_items_.resize(count);
        for (auto& item : vision_items_) {
            ar(item.modality); ar(item.grid); ar(item.patch_begin); ar(item.patch_count);
            ar(item.content_digest); ar(item.timestamps); ar(item.token_spans);
        }
        ar(rewrite_execution_frontiers_);
    }
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(ResidentPrefixIdentity& other) noexcept;
    void append_generated(std::size_t count, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;
    [[nodiscard]] bool equals(const ResidentPrefixIdentity& other) const;
    [[nodiscard]] bool prefix_equals(const ResidentPrefixIdentity& other, std::size_t count) const;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
    std::vector<std::uint32_t> rewrite_execution_frontiers_;
};

// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
class PrefixShortlistDigests {
public:
    template<class Archive> void archive(Archive& ar) { ar(digests_); }
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(PrefixShortlistDigests& other) noexcept;
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0 : digests_.size() - 1U;
    }

    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;

private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  std::span<const TokenId> resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

} // namespace ninfer::models::qwen3_5::detail
