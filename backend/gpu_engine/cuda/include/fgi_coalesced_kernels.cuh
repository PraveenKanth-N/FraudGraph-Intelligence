#ifndef FGI_COALESCED_KERNELS_CUH
#define FGI_COALESCED_KERNELS_CUH

#include "fgi_cuda_common.cuh"

namespace fgi {

/**
 * Warp-per-node coalesced neighbor aggregation kernel.
 *
 * Architecture:
 * - 1 warp (32 threads) is assigned to process one output node i.
 * - For canonical feature dimension D=32, lane t (0 <= t < 32) processes column k = t.
 * - When loading neighbor j's feature vector, the 32 lanes read contiguous addresses
 *   (j * D + 0 ... j * D + 31), generating 100% coalesced 128-byte DRAM transactions.
 * - Arbitrary feature dimensions D are supported via column striding:
 *   for (int k = lane; k < feature_dim; k += 32).
 * - Empty neighborhoods (degree <= 0) strictly return 0.0f across all dimensions.
 * - Max aggregation safely handles all-negative feature values by initializing
 *   from the first neighbor's feature (when degree > 0).
 *
 * Divergence Characteristics:
 * - Intra-warp divergence: ZERO. All 32 lanes in warp i execute the exact same loop
 *   bounds [start_i, end_i) over node i's neighbor list.
 * - Inter-warp divergence: Different warps in the same SM or block process nodes with
 *   different degrees (e.g. d_0 = 1 vs d_1 = 400). Execution time across warps in a block
 *   is bounded by the warp with the largest degree in that block.
 */
__global__ void aggregate_coalesced_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ features,
    float* __restrict__ output,
    int num_nodes,
    int feature_dim,
    AggregationOp op
);

/**
 * Host launch function for warp-coalesced aggregation.
 * @param block_size Number of threads per block (must be a multiple of 32, default: 256 -> 8 warps/block).
 */
void launch_coalesced_aggregation(
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

#endif // FGI_COALESCED_KERNELS_CUH
