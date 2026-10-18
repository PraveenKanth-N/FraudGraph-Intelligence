#ifndef FGI_BASELINE_KERNELS_CUH
#define FGI_BASELINE_KERNELS_CUH

#include "fgi_cuda_common.cuh"

namespace fgi {

/**
 * Baseline CUDA aggregation kernel: 1 thread per output node.
 * Each active thread iterates through the node's neighbor range in col_idx
 * and computes the output feature vector over feature_dim columns.
 *
 * Parameters:
 *   row_ptr:     CSR row offsets array of length num_nodes + 1
 *   col_idx:     CSR neighbor indices array of length num_edges
 *   features:    Input node feature matrix (num_nodes x feature_dim, row-major)
 *   output:      Output aggregated feature matrix (num_nodes x feature_dim, row-major)
 *   num_nodes:   Total number of nodes in graph
 *   feature_dim: Number of features per node
 *   op:          Aggregation operator (OP_SUM, OP_MEAN, OP_MAX)
 */
__global__ void aggregate_baseline_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ features,
    float* __restrict__ output,
    int num_nodes,
    int feature_dim,
    AggregationOp op
);

/**
 * Host-side launcher for baseline CUDA aggregation.
 * Calculates grid dimensions, guards against zero-node launches, and executes the kernel.
 */
void launch_baseline_aggregation(
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_features,
    float* d_output,
    int num_nodes,
    int feature_dim,
    AggregationOp op,
    int block_size = 256,
    cudaStream_t stream = 0
);

} // namespace fgi

#endif // FGI_BASELINE_KERNELS_CUH
