#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

extern "C" void launch_flash_attention_v2(
    cudaStream_t stream,
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int N,
    int D,
    float sm_scale
);

ffi::Error FlashAttentionImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> Q,
    ffi::Buffer<ffi::F32> K,
    ffi::Buffer<ffi::F32> V,
    ffi::ResultBuffer<ffi::F32> O,
    double sm_scale // Accepts Python float (F64)
) {
    auto q_dimensions = Q.dimensions();
    if (q_dimensions.size() != 4) {
        return ffi::Error::InvalidArgument("Expected 4D input tensors [B, H, N, D]");
    }

    int B = q_dimensions[0];
    int H = q_dimensions[1];
    int N = q_dimensions[2];
    int D = q_dimensions[3];

    launch_flash_attention_v2(
        stream,
        Q.typed_data(),
        K.typed_data(),
        V.typed_data(),
        O->typed_data(),
        B, H, N, D,
        static_cast<float>(sm_scale)
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlashAttention,
    FlashAttentionImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>() // Q
        .Arg<ffi::Buffer<ffi::F32>>() // K
        .Arg<ffi::Buffer<ffi::F32>>() // V
        .Ret<ffi::Buffer<ffi::F32>>() // O
        .Attr<double>("sm_scale")     // Match Python float
);