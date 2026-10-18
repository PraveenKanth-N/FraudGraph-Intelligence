#ifndef FGI_CUDA_COMMON_CUH
#define FGI_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

// Robust CUDA API error-checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA ERROR] %s at %s:%d - %s\n",           \
                         #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Check for asynchronous kernel launch errors
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA KERNEL LAUNCH ERROR] at %s:%d - %s\n",\
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace fgi {

// Aggregation operator identifiers matching FGI specifications
enum AggregationOp {
    OP_SUM = 0,
    OP_MEAN = 1,
    OP_MAX = 2
};

inline const char* opToString(AggregationOp op) {
    switch (op) {
        case OP_SUM: return "sum";
        case OP_MEAN: return "mean";
        case OP_MAX: return "max";
        default: return "unknown";
    }
}

inline AggregationOp stringToOp(const std::string& str) {
    if (str == "sum" || str == "SUM") return OP_SUM;
    if (str == "mean" || str == "MEAN") return OP_MEAN;
    if (str == "max" || str == "MAX") return OP_MAX;
    std::fprintf(stderr, "[ERROR] Unknown aggregation operator: %s\n", str.c_str());
    std::exit(EXIT_FAILURE);
}

} // namespace fgi

#endif // FGI_CUDA_COMMON_CUH
