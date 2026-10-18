#include "fgi_baseline_kernels.cuh"

namespace fgi {

__global__ void aggregate_baseline_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ features,
    float* __restrict__ output,
    int num_nodes,
    int feature_dim,
    AggregationOp op
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    int degree = end - start;

    size_t out_offset = (size_t)node * (size_t)feature_dim;

    // Strict zero-sentinel handling for empty neighborhoods (isolated nodes)
    if (degree <= 0) {
        for (int k = 0; k < feature_dim; ++k) {
            output[out_offset + k] = 0.0f;
        }
        return;
    }

    // Aggregation for connected nodes (degree > 0)
    if (op == OP_SUM || op == OP_MEAN) {
        for (int k = 0; k < feature_dim; ++k) {
            float acc = 0.0f;
            for (int p = start; p < end; ++p) {
                int neighbor = col_idx[p];
                size_t feat_idx = (size_t)neighbor * (size_t)feature_dim + (size_t)k;
                acc += features[feat_idx];
            }
            if (op == OP_SUM) {
                output[out_offset + k] = acc;
            } else {
                output[out_offset + k] = acc / (float)degree;
            }
        }
    } else if (op == OP_MAX) {
        for (int k = 0; k < feature_dim; ++k) {
            // Initialize with first neighbor's feature to correctly handle negative-only neighborhoods
            int first_neighbor = col_idx[start];
            size_t first_feat_idx = (size_t)first_neighbor * (size_t)feature_dim + (size_t)k;
            float max_val = features[first_feat_idx];

            for (int p = start + 1; p < end; ++p) {
                int neighbor = col_idx[p];
                size_t feat_idx = (size_t)neighbor * (size_t)feature_dim + (size_t)k;
                float val = features[feat_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
            output[out_offset + k] = max_val;
        }
    }
}

void launch_baseline_aggregation(
    const int* d_row_ptr,
    const int* d_col_idx,
    const float* d_features,
    float* d_output,
    int num_nodes,
    int feature_dim,
    AggregationOp op,
    int block_size,
    cudaStream_t stream
) {
    if (num_nodes <= 0 || feature_dim <= 0) {
        return;
    }

    int grid_size = (num_nodes + block_size - 1) / block_size;
    aggregate_baseline_kernel<<<grid_size, block_size, 0, stream>>>(
        d_row_ptr, d_col_idx, d_features, d_output, num_nodes, feature_dim, op
    );
    CUDA_CHECK_KERNEL();
}

} // namespace fgi
