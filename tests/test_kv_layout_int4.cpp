#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <cstdint>
#include <iostream>

namespace {

int expect(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int expect_bytes(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

// Qwen3.8 27B: 16 full-attention layers plus one MTP layer, 4 KV heads of head_dim 256.
constexpr std::uint32_t kTextLayers = 16;
constexpr std::uint32_t kMtpLayers  = 1;
constexpr std::int32_t kKvHeads     = 4;
constexpr std::int32_t kHeadDim     = 256;
constexpr std::uint32_t kOnePage    = 64;

std::size_t plan_one_page(ninfer::KvCacheStorage storage) {
    ninfer::LayoutBuilder builder;
    const ninfer::targets::qwen3_6::DecoderStateSpec spec{
        .full_attention_layers = kTextLayers,
        .mtp_layers            = kMtpLayers,
        .capacity              = kOnePage,
        .kv_heads              = kKvHeads,
        .attention_head_dim    = kHeadDim,
        .kv_dtype              = ninfer::targets::qwen3_6::kv_storage_dtype(storage),
        .kv_quant_group        = ninfer::targets::qwen3_6::kv_storage_quant_group(storage),
        .enable_mtp            = true,
        .kv_table_rows         = 1,
        .text_physical_page_groups = 1,
        .mtp_physical_page_groups  = 1,
        .linear_attention          = {.layers         = 1,
                                      .conv_channels  = 1,
                                      .conv_width     = 1,
                                      .value_heads    = 1,
                                      .value_head_dim = 1,
                                      .key_head_dim   = 1},
    };
    return ninfer::targets::qwen3_6::plan_decoder_state(builder, spec).kv_payload_bytes();
}

} // namespace

int main() {
    int failures = 0;

    // Per layer and page: two code planes plus, when quantized, two FP16 scale planes.
    // INT8-G64: 2*(256*64*4) + 2*(4*64*4*2)   = 135168
    // INT4-G32: 2*(128*64*4) + 2*(8*64*4*2)   =  73728
    constexpr std::size_t kLayers = kTextLayers + kMtpLayers;
    failures += expect_bytes(plan_one_page(ninfer::KvCacheStorage::Int8Group64),
                             kLayers * 135168ULL, "INT8-G64 bytes per page");
    failures += expect_bytes(plan_one_page(ninfer::KvCacheStorage::Int4Group32),
                             kLayers * 73728ULL, "INT4-G32 bytes per page");
    // BF16 stores no scales: 2*(256*64*4*2) = 262144
    failures += expect_bytes(plan_one_page(ninfer::KvCacheStorage::BFloat16), kLayers * 262144ULL,
                             "BF16 bytes per page");

    failures += expect(ninfer::targets::qwen3_6::kv_storage_dtype(
                           ninfer::KvCacheStorage::Int4Group32) == ninfer::DType::U8,
                       "INT4 codes must live in a U8 plane");
    failures += expect(ninfer::targets::qwen3_6::kv_storage_of(ninfer::DType::U8) ==
                           ninfer::KvCacheStorage::Int4Group32,
                       "U8 KV storage must map back to INT4-G32");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
