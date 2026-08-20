// ninfer::ops - dflash_conv wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/dflash_conv.h"

#include "ops/launcher/dflash_conv.h" // detail::dflash_conv_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* what) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("dflash_conv: ") + what + " must be BF16");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("dflash_conv: ") + what + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("dflash_conv: ") + what + " data must be non-null");
    }
}

} // namespace

void dflash_conv(const Tensor& hidden, const Tensor& base, const Tensor& dynamic,
                 DFlashConvStage stage, bool accumulate, Tensor& y, cudaStream_t stream) {
    if (stage != DFlashConvStage::Prepare && stage != DFlashConvStage::Finish) {
        throw std::invalid_argument("dflash_conv: stage must be Prepare or Finish");
    }
    for (int d = 0; d < 4; ++d) {
        if (hidden.ne[d] != y.ne[d]) {
            throw std::invalid_argument("dflash_conv: hidden/y shapes must match");
        }
        if (hidden.ne[d] < 0) {
            throw std::invalid_argument("dflash_conv: hidden dimensions must be nonnegative");
        }
    }
    if (hidden.ne[3] != 1 || base.ne[3] != 1 || dynamic.ne[3] != 1) {
        throw std::invalid_argument("dflash_conv: inputs are at most three-dimensional");
    }
    const std::int32_t channels = hidden.ne[0];
    const std::int32_t width    = hidden.ne[1];
    const std::int32_t blocks   = hidden.ne[2];
    const std::int32_t taps     = base.ne[1];
    if (base.ne[0] != channels || base.ne[2] != 2 || taps < 1) {
        throw std::invalid_argument("dflash_conv: base must be [D,T,2] with T >= 1");
    }
    if (dynamic.ne[1] != width || dynamic.ne[2] != blocks) {
        throw std::invalid_argument("dflash_conv: dynamic must share the block layout of hidden");
    }
    const std::int32_t stride = 2 * taps;
    if (dynamic.ne[0] <= 0 || (dynamic.ne[0] % stride) != 0) {
        throw std::invalid_argument("dflash_conv: dynamic rows must be a multiple of 2*T");
    }
    const std::int32_t groups = dynamic.ne[0] / stride;
    if (channels <= 0 || (channels % groups) != 0) {
        throw std::invalid_argument("dflash_conv: the group count must divide the channel count");
    }
    if (hidden.numel() == 0) { return; }
    require_bf16(hidden, "hidden");
    require_bf16(base, "base");
    require_bf16(dynamic, "dynamic");
    require_bf16(y, "y");
    if (hidden.data == y.data) {
        throw std::invalid_argument("dflash_conv: hidden and y must not alias");
    }

    detail::dflash_conv_launch(hidden, base, dynamic, stage, accumulate, y, stream);
}

} // namespace ninfer::ops
