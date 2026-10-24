#include "fgi_degree_aware_kernels.cuh"
#include <cstdio>
#include <cmath>

namespace fgi {

// ============================================================================
// KERNEL 1: Standard-Degree Warp Kernel (1 warp per standard node)
// Processes nodes with bounded degree (e.g. 1 <= degree <= THRESHOLD).
// All warps in a thread block retire within a narrow iteration window,
// eliminating inter-warp execution divergence.
// ============================================================================
__global__ void standard_degree_warp_kernel(
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_features,
    float* __restrict__ d_output,
    const int* __restrict__ d_std_nodes,
    int num_std_nodes,
    int feature_dim,
    AggregationOp op
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (global_warp_id >= num_std_nodes) {
        return;
    }

    int node_idx = d_std_nodes[global_warp_id];
    int start = d_row_ptr[node_idx];
    int end = d_row_ptr[node_idx + 1];
    int degree = end - start;
    size_t out_node_offset = (size_t)node_idx * (size_t)feature_dim;

    // Handle empty neighborhood boundary condition (strict zero sentinel)
    if (degree <= 0) {
        for (int k = lane_id; k < feature_dim; k += 32) {
            d_output[out_node_offset + k] = 0.0f;
        }
        return;
    }

    // Column-strided iteration across feature dimensions
    for (int k = lane_id; k < feature_dim; k += 32) {
        if (op == OP_MAX) {
            // First-neighbor initialization for safe negative feature handling
            int first_neighbor = d_col_idx[start];
            float max_val = d_features[(size_t)first_neighbor * (size_t)feature_dim + k];

            #pragma unroll 4
            for (int p = start + 1; p < end; ++p) {
                int neighbor = d_col_idx[p];
                float val = d_features[(size_t)neighbor * (size_t)feature_dim + k];
                max_val = fmaxf(max_val, val);
            }
            d_output[out_node_offset + k] = max_val;
        } else {
            // SUM and MEAN aggregation
            float sum_val = 0.0f;

            #pragma unroll 4
            for (int p = start; p < end; ++p) {
                int neighbor = d_col_idx[p];
                sum_val += d_features[(size_t)neighbor * (size_t)feature_dim + k];
            }

            if (op == OP_MEAN) {
                sum_val = sum_val / (float)degree;
            }
            d_output[out_node_offset + k] = sum_val;
        }
    }
}

// ============================================================================
// KERNEL 2: Hub-Degree Cooperative Block Kernel (1 thread block per hub node)
// An entire thread block (256 threads = 8 warps) cooperatively processes
// a single hub node (degree > THRESHOLD).
// Warps interleave edge reads with 100% 128-byte coalescing, reducing
// sequential iteration depth by 8x.
// ============================================================================
__global__ void hub_degree_block_kernel(
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_features,
    float* __restrict__ d_output,
    const int* __restrict__ d_hub_nodes,
    int num_hub_nodes,
    int feature_dim,
    AggregationOp op
) {
    if (blockIdx.x >= num_hub_nodes) {
        return;
    }

    int node_idx = d_hub_nodes[blockIdx.x];
    int warp_id = threadIdx.x >> 5;  // 0 to 7 (8 warps per 256-thread block)
    int lane_id = threadIdx.x & 31;  // 0 to 31

    int start = d_row_ptr[node_idx];
    int end = d_row_ptr[node_idx + 1];
    int degree = end - start;
    size_t out_node_offset = (size_t)node_idx * (size_t)feature_dim;

    // Shared memory reduction staging buffer: 8 warps x 32 lanes
    __shared__ float s_warp_acc[8][32];

    for (int k = lane_id; k < feature_dim; k += 32) {
        if (op == OP_MAX) {
            float warp_acc = -1.0e38f;
            bool has_val = false;

            // Interleaved edge traversal across the 8 warps
            for (int p = start + warp_id; p < end; p += 8) {
                int neighbor = d_col_idx[p];
                float val = d_features[(size_t)neighbor * (size_t)feature_dim + k];
                if (!has_val) {
                    warp_acc = val;
                    has_val = true;
                } else {
                    warp_acc = fmaxf(warp_acc, val);
                }
            }

            s_warp_acc[warp_id][lane_id] = warp_acc;
            __syncthreads();

            // Warp 0 reduces the 8 warps' partial results for column k
            if (warp_id == 0) {
                float final_max = s_warp_acc[0][lane_id];
                for (int w = 1; w < 8; ++w) {
                    final_max = fmaxf(final_max, s_warp_acc[w][lane_id]);
                }
                d_output[out_node_offset + k] = final_max;
            }
        } else if (op == OP_MEAN) {
            // Cooperative 8-warp interleaved accumulation for MEAN
            float warp_acc = 0.0f;

            for (int p = start + warp_id; p < end; p += 8) {
                int neighbor = d_col_idx[p];
                warp_acc += d_features[(size_t)neighbor * (size_t)feature_dim + k];
            }

            s_warp_acc[warp_id][lane_id] = warp_acc;
            __syncthreads();

            // Warp 0 reduces the 8 warps' partial sums for column k
            if (warp_id == 0) {
                float total_sum = 0.0f;
                for (int w = 0; w < 8; ++w) {
                    total_sum += s_warp_acc[w][lane_id];
                }
                if (degree > 0) {
                    total_sum = total_sum / (float)degree;
                }
                d_output[out_node_offset + k] = total_sum;
            }
        } else {
            // OP_SUM: To guarantee numerical equivalence (atol <= 1e-5) on dense hubs
            // where non-associative float32 addition tree reordering would accumulate
            // drift > 1e-5, Warp 0 performs strictly ordered sequential accumulation.
            if (warp_id == 0) {
                float total_sum = 0.0f;
                #pragma unroll 4
                for (int p = start; p < end; ++p) {
                    int neighbor = d_col_idx[p];
                    total_sum += d_features[(size_t)neighbor * (size_t)feature_dim + k];
                }
                d_output[out_node_offset + k] = total_sum;
            }
        }
        __syncthreads(); // Synchronize before processing next column tile
    }
}

// ============================================================================
// HOST DISPATCH ENTRYPOINT
// ============================================================================
cudaError_t launch_degree_aware_aggregation(
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_features,
    float* d_output,
    const int* d_std_nodes,
    int num_std_nodes,
    const int* d_hub_nodes,
    int num_hub_nodes,
    int num_nodes,
    int feature_dim,
    AggregationOp op,
    int block_size,
    cudaStream_t stream
) {
    if (num_nodes <= 0) {
        return cudaSuccess;
    }

    // 1. Dispatch Standard-Degree Kernel (1 warp per standard node)
    if (num_std_nodes > 0) {
        int warps_per_block = block_size / 32;
        int grid_size = (num_std_nodes + warps_per_block - 1) / warps_per_block;

        standard_degree_warp_kernel<<<grid_size, block_size, 0, stream>>>(
            d_row_ptr, d_col_idx, d_features, d_output,
            d_std_nodes, num_std_nodes, feature_dim, op
        );
        CUDA_CHECK_KERNEL();
    }

    // 2. Dispatch Hub-Degree Cooperative Kernel (1 block = 8 warps per hub node)
    if (num_hub_nodes > 0) {
        int hub_block_size = 256; // Fixed 8 warps per block
        int hub_grid_size = num_hub_nodes;

        hub_degree_block_kernel<<<hub_grid_size, hub_block_size, 0, stream>>>(
            d_row_ptr, d_col_idx, d_features, d_output,
            d_hub_nodes, num_hub_nodes, feature_dim, op
        );
        CUDA_CHECK_KERNEL();
    }

    return cudaGetLastError();
}

} // namespace fgi
