#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

extern "C" void launch_add_one(cudaStream_t stream, const float* in, float* out, int size);

ffi::Error AddOneFFIImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> in,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out
) {
    int size = in.element_count();
    launch_add_one(stream, in.typed_data(), out->typed_data(), size);
    return ffi::Error::Success();
}

// Export raw C symbol for python ctypes discovery
extern "C" {
    XLA_FFI_DEFINE_HANDLER_SYMBOL(
        kAddOne,
        AddOneFFIImpl,
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<ffi::Buffer<ffi::DataType::F32>>()
            .Ret<ffi::Buffer<ffi::DataType::F32>>()
    );
}