#pragma once

#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer {

class DeviceArena;

namespace targets::qwen3_6 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

// A DFlash2 grouped dynamic causal convolution.  Half the kernel is static
// (base_kernel) and half is predicted per token by kernel_projection; the two
// halves along the leading axis serve the prepare and finish steps that
// bracket the attention or MLP block.  Empty on DFlash v1.
struct DFlashConvWeights {
    Tensor base_kernel;
    Weight kernel_projection;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    // Row views of query_key_value. DFlash2 projects the precomputed target context and
    // the drafted block through the same key and value rows.
    Weight query;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
    DFlashConvWeights attention_convolution;
    DFlashConvWeights mlp_convolution;
};

// DFlash2 scores a block of candidates jointly: two rank-256 codebooks give
// each predecessor/successor pair a compatibility term on top of the unary
// logit.  Empty on DFlash v1, which takes an argmax per position instead.
struct DFlashSelectorWeights {
    Tensor hidden_projection;
    Tensor predecessor_codebook;
    Tensor successor_codebook;
};

template <std::size_t Layers>
struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
    DFlashSelectorWeights selector;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
};

} // namespace targets::qwen3_6
} // namespace ninfer
