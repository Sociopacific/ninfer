#include "ops/launcher/dflash_conv.h"

#include "ops/common/math.h"
#include "ops/kernel/dflash_conv.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock          = 256;
constexpr int kMaximumGridDim = 65535;

struct Geometry {
    std::int32_t channels;
    std::int32_t width;
    std::int32_t blocks;
    std::int32_t taps;
    std::int32_t groups;
    std::int32_t channels_per_group;
    std::int32_t dynamic_rows;
    std::int64_t columns;
    std::int64_t total;
};

Geometry describe(const Tensor& hidden, const Tensor& base, const Tensor& dynamic) {
    Geometry geometry{};
    geometry.channels           = hidden.ne[0];
    geometry.width              = hidden.ne[1];
    geometry.blocks             = hidden.ne[2];
    geometry.taps               = base.ne[1];
    geometry.dynamic_rows       = dynamic.ne[0];
    geometry.groups             = geometry.dynamic_rows / (2 * geometry.taps);
    geometry.channels_per_group = geometry.channels / geometry.groups;
    geometry.columns = static_cast<std::int64_t>(geometry.width) * geometry.blocks;
    geometry.total   = geometry.columns * geometry.channels;
    return geometry;
}

template <int Taps>
void launch_packed(const Geometry& geometry, const __nv_bfloat16* hidden,
                   const __nv_bfloat16* base, const __nv_bfloat16* dynamic, __nv_bfloat16* y,
                   bool accumulate, cudaStream_t stream) {
    const std::int32_t packs = geometry.channels / 8;
    const dim3 grid(static_cast<unsigned>(div_up(packs, kBlock)),
                    static_cast<unsigned>(geometry.width),
                    static_cast<unsigned>(geometry.blocks));
    dflash_conv_bf16x8_kernel<kBlock, Taps><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const Bf16x8Pack*>(hidden), reinterpret_cast<const Bf16x8Pack*>(base),
        dynamic, reinterpret_cast<Bf16x8Pack*>(y), packs, geometry.channels_per_group / 8,
        geometry.groups, geometry.width, geometry.dynamic_rows, accumulate);
}

} // namespace

void dflash_conv_launch(const Tensor& hidden, const Tensor& base, const Tensor& dynamic,
                        DFlashConvStage stage, bool accumulate, Tensor& y, cudaStream_t stream) {
    const Geometry geometry = describe(hidden, base, dynamic);
    const auto stage_index  = static_cast<std::int64_t>(stage);
    // `base` is [D,T,2] and `dynamic` rows are [2,T,G]: both stage offsets are a constant shift of
    // the tap index, so the stage selection folds into the base pointers.
    const auto* base_stage = static_cast<const __nv_bfloat16*>(base.data) +
                             stage_index * geometry.taps * geometry.channels;
    const auto* dynamic_stage = static_cast<const __nv_bfloat16*>(dynamic.data) +
                                stage_index * geometry.taps * geometry.groups;
    const auto* hidden_data = static_cast<const __nv_bfloat16*>(hidden.data);
    auto* y_data            = static_cast<__nv_bfloat16*>(y.data);

    const auto addresses = reinterpret_cast<std::uintptr_t>(hidden_data) |
                           reinterpret_cast<std::uintptr_t>(base_stage) |
                           reinterpret_cast<std::uintptr_t>(y_data);
    const bool packed = (geometry.channels % 8) == 0 && (geometry.channels_per_group % 8) == 0 &&
                        (addresses & (alignof(Bf16x8Pack) - 1)) == 0 &&
                        geometry.width <= kMaximumGridDim && geometry.blocks <= kMaximumGridDim;
    if (packed) {
        switch (geometry.taps) {
        case 1:
            launch_packed<1>(geometry, hidden_data, base_stage, dynamic_stage, y_data, accumulate,
                             stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        case 2:
            launch_packed<2>(geometry, hidden_data, base_stage, dynamic_stage, y_data, accumulate,
                             stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        case 3:
            launch_packed<3>(geometry, hidden_data, base_stage, dynamic_stage, y_data, accumulate,
                             stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        case 4:
            launch_packed<4>(geometry, hidden_data, base_stage, dynamic_stage, y_data, accumulate,
                             stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        default: break;
        }
    }
    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(geometry.total, static_cast<std::int64_t>(kBlock)),
        std::numeric_limits<int>::max()));
    dflash_conv_kernel<<<grid, kBlock, 0, stream>>>(
        hidden_data, base_stage, dynamic_stage, y_data, geometry.channels,
        geometry.channels_per_group, geometry.groups, geometry.width, geometry.dynamic_rows,
        geometry.taps, geometry.total, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
