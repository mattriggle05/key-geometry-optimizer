#include "kernels.cuh"
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

// bilinear interpolation into a polar field
// this is the exact function the real project will use
__device__ float lookup_field(float *field, int n_r, int n_theta, float r_max, float dx, float dy) {
    float r = sqrtf(dx * dx + dy * dy);
    float theta = atan2f(dy, dx);
    if (theta < 0.0f) theta += 6.2832f;

    float r_idx = (r / r_max) * (n_r - 1);
    float t_idx = (theta / 6.2832f) * n_theta;

    // clamp r, wrap theta
    if (r_idx >= n_r - 1)
        r_idx = n_r - 1.0001f;
    int r0 = (int)r_idx, r1 = r0 + 1;
    int t0 = (int)t_idx, t1 = (t0 + 1) % n_theta;
    float rf = r_idx - r0;
    float tf = t_idx - t0;

    return field[r0 * n_theta + t0] * (1.0f - rf) * (1.0f - tf) + field[r1 * n_theta + t0] * (rf) * (1.0f - tf) + field[r0 * n_theta + t1] * (1.0f - rf) * (tf) + field[r1 * n_theta + t1] * (rf) * (tf);
}

__global__ void benchmark_kernel(int n_iterations, float *field_data, int n_r, int n_theta, float r_max, ChainResult *results) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // each chain gets its own rng
    curandState rng;
    curand_init(12345ULL, id, 0, &rng);

    float score = 0.0f;
    int accepted = 0;
    int rej_overlap = 0;
    int rej_score = 0;

    // simulate the SA inner loop without actual key state
    // we are testing the speed of the expensive operations
    for (int i = 0; i < n_iterations; i++)
    {

        // random position in a 200mm x 200mm workspace
        float x = (curand_uniform(&rng) - 0.5f) * 200.0f;
        float y = (curand_uniform(&rng) - 0.5f) * 200.0f;

        // overlap check: sqrt against 49 dummy neighbor positions
        // positions are fixed, we just want the sqrt throughput
        bool overlap = false;
        float diam_sq = 19.05f * 19.05f;
        for (int k = 0; k < 49; k++)
        {
            float nx = (k % 7) * 20.0f - 60.0f;
            float ny = (k / 7) * 20.0f - 60.0f;
            float dx = x - nx;
            float dy = y - ny;
            if (dx * dx + dy * dy < diam_sq)
            {
                overlap = true;
                break;
            }
        }

        if (overlap)
        {
            rej_overlap++;
            continue;
        }

        // field lookup: atan2 + sqrt + bilinear interp
        // this is the scoring step
        float effort = lookup_field(field_data, n_r, n_theta, r_max, x, y);

        // accept/reject
        float delta = effort - score;
        float T = 0.5f; // fixed temperature, just testing throughput
        if (delta < 0.0f)
        {
            score = effort;
            accepted++;
        }
        else
        {
            float prob = expf(-delta / T);
            if (curand_uniform(&rng) < prob)
            {
                score = effort;
                accepted++;
            }
            else
            {
                rej_score++;
            }
        }
    }

    results[id].best_score = score;
    results[id].steps_accepted = accepted;
    results[id].steps_rejected_overlap = rej_overlap;
    results[id].steps_rejected_score = rej_score;
}

void launch_benchmark_kernel(
    int n_chains,
    int n_iterations,
    float *d_field_data,
    int n_r,
    int n_theta,
    float r_max,
    ChainResult *d_results)
{
    int threads = 256;
    int blocks = (n_chains + threads - 1) / threads;

    benchmark_kernel<<<blocks, threads>>>(
        n_iterations, d_field_data,
        n_r, n_theta, r_max, d_results);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }
}