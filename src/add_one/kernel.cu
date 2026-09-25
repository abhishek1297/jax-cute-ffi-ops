#include <cuda_runtime.h>

extern "C" __global__ void add_one_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] + 1.0f;
    }
}

extern "C" void launch_add_one(cudaStream_t stream, const float* in, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_one_kernel<<<blocks, threads, 0, stream>>>(in, out, size);
}