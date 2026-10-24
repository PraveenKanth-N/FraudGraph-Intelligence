#pragma once

#include <cuda_runtime.h>
#include "fgi_cuda_common.cuh"

namespace fgi {

// Host-callable dispatch function for Degree-Aware Hybrid Aggregation
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
    int block_size = 256,
    cudaStream_t stream = 0
);

} // namespace fgi
