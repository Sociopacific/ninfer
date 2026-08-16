#pragma once

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

enum class ProcessorErrorKind {
    BudgetExceeded,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

private:
    ProcessorErrorKind kind_;
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t prompt_tokens     = 0;
    std::size_t patch_bytes       = 0;

    [[nodiscard]] std::string summary() const;
};

// One media part after decode, resize and patchification.
struct PreparedMedia {
    VisionItem item;
    std::vector<float> patches;
};

// LRU over preprocessed media, keyed by the digest of the source bytes.
//
// Decoding one screenshot costs on the order of a second of CPU, and an agentic
// conversation resends every image it has ever read on every turn -- so without
// this the cost is paid again on each request, long after the KV cache stopped
// having to prefill any of those tokens. Methods are const and internally
// locked: the cache lives in a shared const Frontend::Impl.
class MediaCache {
public:
    struct Key {
        std::array<std::uint8_t, 32> digest{};
        // Covers the processor knobs that change output geometry, so a
        // reconfigured server cannot be served stale patches.
        std::uint64_t fingerprint = 0;

        [[nodiscard]] bool operator==(const Key& other) const noexcept {
            return digest == other.digest && fingerprint == other.fingerprint;
        }
    };

    explicit MediaCache(std::size_t capacity_bytes) noexcept : capacity_bytes_(capacity_bytes) {}

    [[nodiscard]] bool lookup(const Key& key, PreparedMedia& out) const;
    void store(const Key& key, const PreparedMedia& value) const;

private:
    struct Entry {
        Key key;
        PreparedMedia value;
        std::size_t bytes = 0;
    };

    struct KeyHash {
        [[nodiscard]] std::size_t operator()(const Key& key) const noexcept;
    };

    std::size_t capacity_bytes_;
    mutable std::mutex mutex_;
    // Front is the most recently used entry.
    mutable std::list<Entry> entries_;
    mutable std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> index_;
    mutable std::size_t used_bytes_ = 0;
};

struct ProcessorOptions {
    std::uint64_t image_min_pixels         = 32ULL * 32ULL;
    std::uint64_t image_max_pixels         = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels         = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels         = 4ULL * 1024ULL * 1024ULL;
    std::size_t max_media_bytes            = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    // These three bound the request, not one item, and they are load-bearing: the
    // vision workspace is sized once per Program from kFrontendMergedLimit, which
    // is this same 32768, and max_raw_patches is the same ceiling counted in
    // patches. Relaxing them does not fail cleanly -- the encode runs past a
    // buffer that was never sized for the sum and corrupts the heap. The
    // per-item check in request_plan_impl.h bounds one item and does not cover
    // this. To fit more images in a request, lower the cost of an image
    // (kImagePixelPolicyCeiling in frontend.cpp), never raise these.
    std::size_t max_media_items            = 16;
    std::uint64_t max_raw_patches          = 131'072;
    std::uint64_t max_vision_tokens        = 32'768;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // Row-major [sum(raw_patches), 1536], in the exact merger-friendly order.
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    PreprocessStats stats;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

struct EncodedChat {
    std::vector<int> input_ids;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
};

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered);

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options = {}, const MediaCache* media_cache = nullptr);

    ProcessedInput process(const std::vector<ChatMessage>& messages,
                           ChatRenderOptions render_options = {}) const;

private:
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
    const MediaCache* media_cache_ = nullptr;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
