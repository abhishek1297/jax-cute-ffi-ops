#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

extern "C" void launch_add_one(cudaStream_t stream, const float* in, float* out, int size);

ffi::Error AddOneImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> in,
    ffi::ResultBuffer<ffi::F32> out
) {
    int size = in.element_count();
    launch_add_one(stream, in.typed_data(), out->typed_data(), size);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AddOne,
    AddOneImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);