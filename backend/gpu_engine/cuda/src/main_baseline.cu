#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "fgi_cuda_common.cuh"
#include "fgi_baseline_kernels.cuh"

// Structure to hold binary CSR graph arrays on host
struct HostCSRGraph {
    int num_nodes = 0;
    int num_edges = 0;
    int feature_dim = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> features;
};

static std::string format_with_commas(std::uint64_t value) {
    std::string s = std::to_string(value);
    int insert_pos = (int)s.length() - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

// Reads raw binary files dumped by the python preprocessor
bool load_raw_csr(const std::string& dir_path, HostCSRGraph& graph) {
    std::string meta_path = dir_path + "/meta.txt";
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        std::fprintf(stderr, "[ERROR] Could not open metadata file: %s\n", meta_path.c_str());
        return false;
    }

    meta_file >> graph.num_nodes >> graph.num_edges >> graph.feature_dim;
    meta_file.close();

    // Read row_ptr
    std::string row_path = dir_path + "/row_ptr.bin";
    std::ifstream row_file(row_path, std::ios::binary);
    if (!row_file.is_open()) {
        std::fprintf(stderr, "[ERROR] Could not open: %s\n", row_path.c_str());
        return false;
    }
    graph.row_ptr.resize(graph.num_nodes + 1);
    row_file.read(reinterpret_cast<char*>(graph.row_ptr.data()), (graph.num_nodes + 1) * sizeof(int));
    row_file.close();

    // Read col_idx
    std::string col_path = dir_path + "/col_idx.bin";
    std::ifstream col_file(col_path, std::ios::binary);
    if (!col_file.is_open()) {
        std::fprintf(stderr, "[ERROR] Could not open: %s\n", col_path.c_str());
        return false;
    }
    graph.col_idx.resize(graph.num_edges);
    col_file.read(reinterpret_cast<char*>(graph.col_idx.data()), graph.num_edges * sizeof(int));
    col_file.close();

    // Read features
    std::string feat_path = dir_path + "/features.bin";
    std::ifstream feat_file(feat_path, std::ios::binary);
    if (!feat_file.is_open()) {
        std::fprintf(stderr, "[ERROR] Could not open: %s\n", feat_path.c_str());
        return false;
    }
    size_t feat_count = (size_t)graph.num_nodes * (size_t)graph.feature_dim;
    graph.features.resize(feat_count);
    feat_file.read(reinterpret_cast<char*>(graph.features.data()), feat_count * sizeof(float));
    feat_file.close();

    return true;
}

int main(int argc, char** argv) {
    std::string graph_dir = "";
    std::string op_str = "mean";
    std::string ref_out_path = "";
    std::string json_export_path = "";
    std::string out_bin_path = "";
    int warmup_iters = 5;
    int repeat_iters = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc) graph_dir = argv[++i];
        else if (arg == "--op" && i + 1 < argc) op_str = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup_iters = std::atoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc) repeat_iters = std::atoi(argv[++i]);
        else if (arg == "--validate" && i + 1 < argc) ref_out_path = argv[++i];
        else if (arg == "--export-json" && i + 1 < argc) json_export_path = argv[++i];
        else if (arg == "--out-bin" && i + 1 < argc) out_bin_path = argv[++i];
    }

    if (graph_dir.empty()) {
        std::fprintf(stderr, "Usage: %s --graph <raw_csr_dir> [--op mean|sum|max] [--warmup 5] [--repeat 20] [--validate <cpu_ref.bin>] [--export-json <out.json>] [--out-bin <out.bin>]\n", argv[0]);
        return 1;
    }

    fgi::AggregationOp op = fgi::stringToOp(op_str);

    // 1. Query GPU Device Properties
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::printf("================================================================================\n");
    std::printf(" FRAUDGRAPH INTELLIGENCE — CUDA BASELINE RUNNER\n");
    std::printf(" GPU: %s | CC: %d.%d | VRAM: %.1f MB | SM Count: %d\n",
                prop.name, prop.major, prop.minor,
                (float)prop.totalGlobalMem / (1024.0f * 1024.0f), prop.multiProcessorCount);
    std::printf(" Operator: %s | Warmup: %d | Repetitions: %d\n",
                fgi::opToString(op), warmup_iters, repeat_iters);
    std::printf("================================================================================\n");

    // 2. Load CSR Graph
    HostCSRGraph graph;
    if (!load_raw_csr(graph_dir, graph)) {
        return 1;
    }
    std::printf("[+] Loaded CSR Graph: %d nodes, %d edges, %d features\n",
                graph.num_nodes, graph.num_edges, graph.feature_dim);

    // 3. Compute 64-bit Memory Sizes
    size_t bytes_row = (size_t)(graph.num_nodes + 1) * sizeof(int);
    size_t bytes_col = (size_t)graph.num_edges * sizeof(int);
    size_t bytes_feat = (size_t)graph.num_nodes * (size_t)graph.feature_dim * sizeof(float);
    size_t bytes_out = bytes_feat;

    // Check device memory availability
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t total_required = bytes_row + bytes_col + bytes_feat + bytes_out;
    if (total_required > free_mem) {
        std::fprintf(stderr, "[ERROR] Required memory (%zu bytes) exceeds free VRAM (%zu bytes)\n",
                     total_required, free_mem);
        return 2;
    }

    // 4. Allocate Device Memory
    int *d_row_ptr = nullptr, *d_col_idx = nullptr;
    float *d_features = nullptr, *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, bytes_row));
    CUDA_CHECK(cudaMalloc(&d_col_idx, bytes_col));
    CUDA_CHECK(cudaMalloc(&d_features, bytes_feat));
    CUDA_CHECK(cudaMalloc(&d_output, bytes_out));

    // 5. Timed H2D Transfers
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), bytes_row, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx.data(), bytes_col, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_features, graph.features.data(), bytes_feat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_event, stop_event));

    // 6. Warm-Up Iterations
    for (int w = 0; w < warmup_iters; ++w) {
        fgi::launch_baseline_aggregation(
            d_row_ptr, d_col_idx, d_features, d_output,
            graph.num_nodes, graph.feature_dim, op, 256, 0
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Measured Kernel Iterations
    std::vector<float> kernel_trials_ms;
    kernel_trials_ms.reserve(repeat_iters);

    for (int r = 0; r < repeat_iters; ++r) {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
        fgi::launch_baseline_aggregation(
            d_row_ptr, d_col_idx, d_features, d_output,
            graph.num_nodes, graph.feature_dim, op, 256, 0
        );
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start_event, stop_event));
        kernel_trials_ms.push_back(iter_ms);
    }

    // 8. Timed D2H Transfer
    std::vector<float> h_output((size_t)graph.num_nodes * (size_t)graph.feature_dim, 0.0f);
    float d2h_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start_event, 0));
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_event, stop_event));

    // 9. Calculate Statistical Metrics
    std::vector<float> sorted_trials = kernel_trials_ms;
    std::sort(sorted_trials.begin(), sorted_trials.end());

    float sum_trials = std::accumulate(kernel_trials_ms.begin(), kernel_trials_ms.end(), 0.0f);
    float mean_kernel_ms = sum_trials / (float)repeat_iters;

    float variance = 0.0f;
    for (float t : kernel_trials_ms) {
        variance += (t - mean_kernel_ms) * (t - mean_kernel_ms);
    }
    float std_kernel_ms = (repeat_iters > 1) ? std::sqrt(variance / (float)(repeat_iters - 1)) : 0.0f;
    float p50_kernel_ms = sorted_trials[repeat_iters / 2];
    int p95_idx = std::min((int)(repeat_iters * 0.95), repeat_iters - 1);
    float p95_kernel_ms = sorted_trials[p95_idx];
    float min_kernel_ms = sorted_trials.front();
    float max_kernel_ms = sorted_trials.back();

    float e2e_ms = h2d_ms + mean_kernel_ms + d2h_ms;
    float edges_per_sec = (mean_kernel_ms > 0.0f) ? ((float)graph.num_edges / (mean_kernel_ms * 1e-3f)) : 0.0f;

    // Compulsory physical DRAM traffic: Reading row_ptr, col_idx, unique node features once + writing output once
    size_t bytes_compulsory = bytes_row + bytes_col + ((size_t)graph.num_nodes * (size_t)graph.feature_dim * sizeof(float)) + bytes_out;
    float bw_compulsory_gbps = (mean_kernel_ms > 0.0f) ? ((float)bytes_compulsory / (mean_kernel_ms * 1e-3f * 1e9f)) : 0.0f;

    // Algorithmic memory traffic: Counts repeated feature reads for every directed edge (assuming zero cache reuse)
    size_t bytes_algorithmic = bytes_row + bytes_col + ((size_t)graph.num_edges * (size_t)graph.feature_dim * sizeof(float)) + bytes_out;
    float bw_algo_gbps = (mean_kernel_ms > 0.0f) ? ((float)bytes_algorithmic / (mean_kernel_ms * 1e-3f * 1e9f)) : 0.0f;

    std::printf("[+] Benchmarks Recorded:\n");
    std::printf("    H2D Transfer:      %8.3f ms\n", h2d_ms);
    std::printf("    Kernel Mean:       %8.3f ms +/- %6.3f ms (p50: %.3f ms, p95: %.3f ms)\n",
                mean_kernel_ms, std_kernel_ms, p50_kernel_ms, p95_kernel_ms);
    std::printf("    D2H Transfer:      %8.3f ms\n", d2h_ms);
    std::printf("    End-to-End Mean:   %8.3f ms\n", e2e_ms);
    std::printf("    Kernel Throughput: %15s edges/sec\n", format_with_commas((std::uint64_t)edges_per_sec).c_str());
    std::printf("    Est. Compulsory BW:%8.2f GB/s (analytical unique footprint: %.2f MB)\n", bw_compulsory_gbps, (float)bytes_compulsory / 1.0e6f);
    std::printf("    Est. Algorithmic BW:%7.2f GB/s (analytical un-cached traffic: %.2f MB)\n", bw_algo_gbps, (float)bytes_algorithmic / 1.0e6f);

    // 10. Numerical Correctness Verification
    std::string verif_status = "NOT_VALIDATED";
    float max_abs_err = 0.0f;
    bool nan_detected = false;

    // Check for NaNs or Infs
    for (float val : h_output) {
        if (std::isnan(val) || std::isinf(val)) {
            nan_detected = true;
            verif_status = "FAILED_NAN_OR_INF";
            break;
        }
    }

    if (!nan_detected && !ref_out_path.empty()) {
        std::ifstream ref_file(ref_out_path, std::ios::binary);
        if (ref_file.is_open()) {
            std::vector<float> ref_output(h_output.size());
            ref_file.read(reinterpret_cast<char*>(ref_output.data()), h_output.size() * sizeof(float));
            ref_file.close();

            for (size_t idx = 0; idx < h_output.size(); ++idx) {
                float diff = std::fabs(h_output[idx] - ref_output[idx]);
                if (diff > max_abs_err) {
                    max_abs_err = diff;
                }
            }

            if (max_abs_err <= 1.0e-5f) {
                verif_status = "PASSED";
            } else {
                verif_status = "FAILED_TOLERANCE";
            }
            std::printf("[+] Validation vs CPU Reference: %s (Max Abs Error: %.2e, Tol: 1.0e-5)\n",
                        verif_status.c_str(), max_abs_err);
        } else {
            std::fprintf(stderr, "[WARNING] Could not open CPU reference file: %s\n", ref_out_path.c_str());
        }
    }

    // 11. Optionally save output binary
    if (!out_bin_path.empty()) {
        std::ofstream out_file(out_bin_path, std::ios::binary);
        if (out_file.is_open()) {
            out_file.write(reinterpret_cast<const char*>(h_output.data()), bytes_out);
            out_file.close();
            std::printf("[+] Saved GPU output tensor: %s\n", out_bin_path.c_str());
        }
    }

    // 12. Export JSON Results
    if (!json_export_path.empty()) {
        std::ofstream json_file(json_export_path);
        if (json_file.is_open()) {
            json_file << "{\n";
            json_file << "  \"schema_version\": \"1.0.0\",\n";
            json_file << "  \"experiment_id\": \"EXP-GPU-BASELINE\",\n";
            json_file << "  \"measurement_state\": \"MEASURED\",\n";
            json_file << "  \"hardware\": {\n";
            json_file << "    \"device_name\": \"" << prop.name << "\",\n";
            json_file << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
            json_file << "    \"total_vram_mb\": " << ((float)prop.totalGlobalMem / (1024.0f * 1024.0f)) << ",\n";
            json_file << "    \"sm_count\": " << prop.multiProcessorCount << "\n";
            json_file << "  },\n";
            json_file << "  \"graph_configuration\": {\n";
            json_file << "    \"num_nodes\": " << graph.num_nodes << ",\n";
            json_file << "    \"num_edges\": " << graph.num_edges << ",\n";
            json_file << "    \"feature_dim\": " << graph.feature_dim << "\n";
            json_file << "  },\n";
            json_file << "  \"benchmark_parameters\": {\n";
            json_file << "    \"implementation\": \"cuda_baseline_thread_per_node\",\n";
            json_file << "    \"operator\": \"" << fgi::opToString(op) << "\",\n";
            json_file << "    \"warmup_iterations\": " << warmup_iters << ",\n";
            json_file << "    \"recorded_repetitions\": " << repeat_iters << "\n";
            json_file << "  },\n";
            json_file << "  \"timing_results_ms\": {\n";
            json_file << "    \"h2d_transfer_mean\": " << h2d_ms << ",\n";
            json_file << "    \"kernel_mean\": " << mean_kernel_ms << ",\n";
            json_file << "    \"kernel_std\": " << std_kernel_ms << ",\n";
            json_file << "    \"kernel_p50\": " << p50_kernel_ms << ",\n";
            json_file << "    \"kernel_p95\": " << p95_kernel_ms << ",\n";
            json_file << "    \"d2h_transfer_mean\": " << d2h_ms << ",\n";
            json_file << "    \"end_to_end_mean\": " << e2e_ms << "\n";
            json_file << "  },\n";
            json_file << "  \"performance_metrics\": {\n";
            json_file << "    \"edges_per_second\": " << edges_per_sec << ",\n";
            json_file << "    \"estimated_compulsory_bandwidth_gbps\": " << bw_compulsory_gbps << ",\n";
            json_file << "    \"estimated_algorithmic_bandwidth_gbps\": " << bw_algo_gbps << ",\n";
            json_file << "    \"effective_bandwidth_gbps\": " << bw_compulsory_gbps << "\n";
            json_file << "  },\n";
            json_file << "  \"numerical_verification\": {\n";
            json_file << "    \"status\": \"" << verif_status << "\",\n";
            json_file << "    \"max_absolute_error\": " << max_abs_err << ",\n";
            json_file << "    \"tolerance_threshold\": 1e-05,\n";
            json_file << "    \"nan_detected\": " << (nan_detected ? "true" : "false") << "\n";
            json_file << "  }\n";
            json_file << "}\n";
            json_file.close();
            std::printf("[+] Exported JSON Results: %s\n", json_export_path.c_str());
        }
    }

    // 13. Cleanup
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_features));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return (verif_status == "FAILED_TOLERANCE" || verif_status == "FAILED_NAN_OR_INF") ? 3 : 0;
}
