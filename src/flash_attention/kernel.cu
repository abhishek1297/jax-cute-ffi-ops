#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cfloat>

using namespace cute;

template <typename T, int kBlockM, int kBlockN, int kHeadDim>
__global__ void flash_attention_v2_cute_kernel(
    const T* __restrict__ Q_ptr,
    const T* __restrict__ K_ptr,
    const T* __restrict__ V_ptr,
    T* __restrict__ O_ptr,
    int seq_len,
    float sm_scale
) {
    // 1. Thread and Grid Mapping
    // Grid structure: dim3 grid( (N + kBlockM - 1) / kBlockM, H, B )
    const int m_block_idx = blockIdx.x; // Block along Sequence Length N
    const int hidx        = blockIdx.y; // Head Index H
    const int bidx        = blockIdx.z; // Batch Index B
    const int tid         = threadIdx.x;

    // 2. Base Pointers for (Batch, Head)
    int batch_head_offset = (bidx * gridDim.y + hidx) * seq_len * kHeadDim;

    auto shape_matrix  = make_shape(seq_len, Int<kHeadDim>{});
    auto stride_matrix = make_stride(Int<kHeadDim>{}, Int<1>{});

    Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr + batch_head_offset), shape_matrix, stride_matrix);
    Tensor gK = make_tensor(make_gmem_ptr(K_ptr + batch_head_offset), shape_matrix, stride_matrix);
    Tensor gV = make_tensor(make_gmem_ptr(V_ptr + batch_head_offset), shape_matrix, stride_matrix);
    Tensor gO = make_tensor(make_gmem_ptr(O_ptr + batch_head_offset), shape_matrix, stride_matrix);

    // 3. Shared Memory
    __shared__ T smem_q[kBlockM * kHeadDim];
    __shared__ T smem_k[kBlockN * kHeadDim];
    __shared__ T smem_v[kBlockN * kHeadDim];

    Tensor sQ = make_tensor(make_smem_ptr(smem_q), make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor sK = make_tensor(make_smem_ptr(smem_k), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor sV = make_tensor(make_smem_ptr(smem_v), make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));

    // Local Registers for Online Softmax
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc[kHeadDim] = {0.0f};

    // 4. Vectorized Tiled Copy (128 threads -> 16x8 layout)
    auto gmem_thr_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<4>{}))
    );

    auto thr_copy = gmem_thr_copy.get_slice(tid);

    // Tile Global Q for current Block Index m_block_idx
    Tensor gQ_block = local_tile(gQ, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block_idx, 0));
    Tensor tQgQ = thr_copy.partition_S(gQ_block);
    Tensor tQsQ = thr_copy.partition_D(sQ);

    // Load Q tile to Shared Memory
    copy(gmem_thr_copy, tQgQ, tQsQ);
    __syncthreads();

    // 5. Main Loop over K and V Tiles
    int n_tiles = (seq_len + kBlockN - 1) / kBlockN;
    for (int j = 0; j < n_tiles; ++j) {
        Tensor gK_block = local_tile(gK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(j, 0));
        Tensor gV_block = local_tile(gV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(j, 0));

        Tensor tKgK = thr_copy.partition_S(gK_block);
        Tensor tKsK = thr_copy.partition_D(sK);
        Tensor tVgV = thr_copy.partition_S(gV_block);
        Tensor tVsV = thr_copy.partition_D(sV);

        copy(gmem_thr_copy, tKgK, tKsK);
        copy(gmem_thr_copy, tVgV, tVsV);
        __syncthreads();

        // 6. Compute Q @ K^T and Online Softmax update
        // Assign 1 row of sQ per thread (threads 0..kBlockM-1)
        int row_in_block = tid;
        if (row_in_block < kBlockM && (m_block_idx * kBlockM + row_in_block) < seq_len) {
            float row_max = -FLT_MAX;
            float S_row[kBlockN];

            for (int col = 0; col < kBlockN; ++col) {
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    score += static_cast<float>(sQ(row_in_block, d)) * static_cast<float>(sK(col, d));
                }
                score *= sm_scale;
                S_row[col] = score;
                row_max = fmaxf(row_max, score);
            }

            // Online Softmax Scaling Math
            float m_next = fmaxf(m_i, row_max);
            float alpha  = expf(m_i - m_next);

            for (int d = 0; d < kHeadDim; ++d) {
                acc[d] *= alpha;
            }

            float l_next = l_i * alpha;
            for (int col = 0; col < kBlockN; ++col) {
                float p_val = expf(S_row[col] - m_next);
                l_next += p_val;
                for (int d = 0; d < kHeadDim; ++d) {
                    acc[d] += p_val * static_cast<float>(sV(col, d));
                }
            }

            m_i = m_next;
            l_i = l_next;
        }
        __syncthreads();
    }

    // 7. Store Output back to Global Memory
    int row_in_block = tid;
    int global_row   = m_block_idx * kBlockM + row_in_block;

    if (row_in_block < kBlockM && global_row < seq_len) {
        Tensor gO_block = local_tile(gO, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block_idx, 0));
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

        for (int d = 0; d < kHeadDim; ++d) {
            gO_block(row_in_block, d) = static_cast<T>(acc[d] * inv_l);
        }
    }
}

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
) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 64;

    int grid_x = (N + kBlockM - 1) / kBlockM;
    dim3 grid(grid_x, H, B);
    dim3 block(128);

    flash_attention_v2_cute_kernel<float, kBlockM, kBlockN, kHeadDim><<<grid, block, 0, stream>>>(
        Q, K, V, O, N, sm_scale
    );
}