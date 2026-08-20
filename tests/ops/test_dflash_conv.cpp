#include "ninfer/ops/dflash_conv.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// A stored BF16 result carries at most half an ulp of its own magnitude, so the criterion is
// relative. The absolute floor sits far below BF16 resolution for any normal magnitude and only
// covers the FP32 accumulator when two taps cancel almost exactly.
constexpr PointwiseCriterion dflash_conv_criterion() {
    return {/*absolute*/ 1.0e-5, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

struct Shape {
    std::int32_t channels;
    std::int32_t width;
    std::int32_t blocks;
    std::int32_t taps;
    std::int32_t group_size;
};

std::vector<double> dflash_conv_oracle(const Shape& shape, const std::vector<float>& hidden,
                                       const std::vector<float>& base,
                                       const std::vector<float>& dynamic,
                                       const std::vector<float>& initial, std::int32_t stage,
                                       bool accumulate) {
    const std::int32_t groups       = shape.channels / shape.group_size;
    const std::int32_t dynamic_rows = 2 * shape.taps * groups;
    std::vector<double> expected(hidden.size());
    for (std::int32_t block = 0; block < shape.blocks; ++block) {
        for (std::int32_t position = 0; position < shape.width; ++position) {
            const std::int64_t column = static_cast<std::int64_t>(block) * shape.width + position;
            for (std::int32_t channel = 0; channel < shape.channels; ++channel) {
                const std::int64_t index = column * shape.channels + channel;
                const std::int32_t group = channel / shape.group_size;
                double sum               = 0.0;
                for (std::int32_t tap = 0; tap < shape.taps; ++tap) {
                    if (position < tap) { break; }
                    const std::int64_t base_index =
                        static_cast<std::int64_t>(stage * shape.taps + tap) * shape.channels +
                        channel;
                    const std::int64_t dynamic_index =
                        column * dynamic_rows +
                        static_cast<std::int64_t>((stage * shape.taps + tap) * groups + group);
                    const std::int64_t hidden_index = (column - tap) * shape.channels + channel;
                    sum += (static_cast<double>(base[static_cast<std::size_t>(base_index)]) +
                            static_cast<double>(dynamic[static_cast<std::size_t>(dynamic_index)])) *
                           static_cast<double>(hidden[static_cast<std::size_t>(hidden_index)]);
                }
                if (accumulate) { sum += static_cast<double>(initial[static_cast<std::size_t>(index)]); }
                expected[static_cast<std::size_t>(index)] = sum;
            }
        }
    }
    return expected;
}

int run_case(const char* label, const Shape& shape, ops::DFlashConvStage stage, bool accumulate,
             std::uint32_t seed) {
    const std::int32_t groups       = shape.channels / shape.group_size;
    const std::int32_t dynamic_rows = 2 * shape.taps * groups;
    const std::size_t columns =
        static_cast<std::size_t>(shape.width) * static_cast<std::size_t>(shape.blocks);
    const std::size_t hidden_count  = columns * static_cast<std::size_t>(shape.channels);
    const std::size_t base_count    = static_cast<std::size_t>(2 * shape.taps) *
                                   static_cast<std::size_t>(shape.channels);
    const std::size_t dynamic_count = columns * static_cast<std::size_t>(dynamic_rows);

    std::vector<float> hidden(hidden_count), base(base_count), dynamic(dynamic_count),
        initial(hidden_count);
    fill_uniform(hidden, seed, -4.0f, 4.0f);
    fill_uniform(base, seed + 1, -1.0f, 1.0f);
    fill_uniform(dynamic, seed + 2, -1.0f, 1.0f);
    fill_uniform(initial, seed + 3, -4.0f, 4.0f);
    round_to_bf16(hidden);
    round_to_bf16(base);
    round_to_bf16(dynamic);
    round_to_bf16(initial);

    const auto expected = dflash_conv_oracle(shape, hidden, base, dynamic, initial,
                                             static_cast<std::int32_t>(stage), accumulate);
    const auto hidden_bits  = encode_bf16(hidden);
    const auto base_bits    = encode_bf16(base);
    const auto dynamic_bits = encode_bf16(dynamic);
    const auto initial_bits = encode_bf16(initial);

    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(dynamic_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_y(initial_bits.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_base.copy_from_host(base_bits.data(), device_base.bytes());
    device_dynamic.copy_from_host(dynamic_bits.data(), device_dynamic.bytes());
    device_y.copy_from_host(initial_bits.data(), device_y.bytes());

    Tensor hidden_tensor(device_hidden.data(), DType::BF16,
                         {shape.channels, shape.width, shape.blocks});
    Tensor base_tensor(device_base.data(), DType::BF16, {shape.channels, shape.taps, 2});
    Tensor dynamic_tensor(device_dynamic.data(), DType::BF16,
                          {dynamic_rows, shape.width, shape.blocks});
    Tensor y_tensor(device_y.data(), DType::BF16, {shape.channels, shape.width, shape.blocks});
    ops::dflash_conv(hidden_tensor, base_tensor, dynamic_tensor, stage, accumulate, y_tensor,
                     nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_y.data(), hidden_count),
                                    expected, dflash_conv_criterion());
    failures += verify_exact("dflash_conv hidden unchanged",
                             from_device<std::uint16_t>(device_hidden.data(), hidden_bits.size()),
                             hidden_bits);
    failures += verify_exact("dflash_conv base unchanged",
                             from_device<std::uint16_t>(device_base.data(), base_bits.size()),
                             base_bits);
    failures += device_hidden.verify_guards("dflash_conv hidden");
    failures += device_base.verify_guards("dflash_conv base");
    failures += device_dynamic.verify_guards("dflash_conv dynamic");
    failures += device_y.verify_guards("dflash_conv y");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // The registered DFlash2 draft geometry: 5120 channels, groups of 16, two taps, one block of
    // eight positions per lane.
    failures += run_case("dflash_conv [5120,8,1] prepare", {5120, 8, 1, 2, 16},
                         ops::DFlashConvStage::Prepare, false, 101u);
    failures += run_case("dflash_conv [5120,8,4] finish+residual", {5120, 8, 4, 2, 16},
                         ops::DFlashConvStage::Finish, true, 201u);
    // A single position leaves only the zeroth tap live.
    failures += run_case("dflash_conv [5120,1,2] prepare", {5120, 1, 2, 2, 16},
                         ops::DFlashConvStage::Prepare, false, 301u);
    // Groups narrower than a pack fall back to scalar indexing.
    failures += run_case("dflash_conv [96,5,3] scalar", {96, 5, 3, 2, 4},
                         ops::DFlashConvStage::Finish, true, 401u);
    // Three taps exercise the deeper causal ramp.
    failures += run_case("dflash_conv [512,6,2] three taps", {512, 6, 2, 3, 8},
                         ops::DFlashConvStage::Prepare, false, 501u);
    std::cout << (failures ? "FAIL" : "OK") << " dflash_conv\n";
    return failures ? 1 : 0;
}
