#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "fields.h"
#include "kernels.cuh"
#include <cuda_runtime.h>

int main()
{

    // --- config ---
    const int N_CHAINS = 10000;
    const int N_ITERATIONS = 100000; // lower than final, just benchmarking
    const int N_R = 32;
    const int N_THETA = 64;
    const float R_MAX = 120.0f;

    // --- print device info ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "CUDA cores (SMs x 128): " << prop.multiProcessorCount * 128 << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";

    // --- build dummy effort field on CPU ---
    FingerField field = make_dummy_field(0.0f, 0.0f, N_R, N_THETA, R_MAX);
    int field_size = N_R * N_THETA;

    // --- copy field to GPU ---
    float *d_field;
    cudaMalloc(&d_field, field_size * sizeof(float));
    cudaMemcpy(d_field, field.data,
               field_size * sizeof(float), cudaMemcpyHostToDevice);

    // --- allocate results on GPU ---
    ChainResult *d_results;
    cudaMalloc(&d_results, N_CHAINS * sizeof(ChainResult));

    // --- run and time it ---
    std::cout << "Running " << N_CHAINS << " chains x "
              << N_ITERATIONS << " iterations...\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    launch_benchmark_kernel(
        N_CHAINS, N_ITERATIONS,
        d_field, N_R, N_THETA, R_MAX,
        d_results);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- copy results back ---
    std::vector<ChainResult> results(N_CHAINS);
    cudaMemcpy(results.data(), d_results,
               N_CHAINS * sizeof(ChainResult), cudaMemcpyDeviceToHost);

    // --- print summary ---
    double avg_accepted = 0, avg_rej_overlap = 0, avg_rej_score = 0;
    for (auto &r : results)
    {
        avg_accepted += r.steps_accepted;
        avg_rej_overlap += r.steps_rejected_overlap;
        avg_rej_score += r.steps_rejected_score;
    }
    avg_accepted /= N_CHAINS;
    avg_rej_overlap /= N_CHAINS;
    avg_rej_score /= N_CHAINS;

    std::cout << "\nResults (averaged across chains):\n";
    std::cout << "  Accepted:          " << (int)avg_accepted << "\n";
    std::cout << "  Rejected (overlap):" << (int)avg_rej_overlap << "\n";
    std::cout << "  Rejected (score):  " << (int)avg_rej_score << "\n";
    std::cout << "\nTotal wall time: " << ms << " ms\n";
    std::cout << "Total iterations: "
              << (long long)N_CHAINS * N_ITERATIONS << "\n";
    std::cout << "Throughput: "
              << (long long)N_CHAINS * N_ITERATIONS / (ms / 1000.0) / 1e9
              << " billion iterations/sec\n";

    // --- cleanup ---
    cudaFree(d_field);
    cudaFree(d_results);
    free_field(field);

    return 0;
}