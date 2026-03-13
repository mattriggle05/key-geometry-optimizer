#pragma once
#include "fields.h"

// result written back for one chain
struct ChainResult
{
    float best_score;
    int steps_accepted;
    int steps_rejected_overlap;
    int steps_rejected_score;
};

// tests three things:
//   1. parallel RNG across thousands of chains
//   2. sqrt + atan2 in the inner loop (the expensive ops)
//   3. bilinear interpolation into a 2D field
void launch_benchmark_kernel(
    int n_chains,
    int n_iterations,
    float *d_field_data, // effort field already on GPU
    int n_r,
    int n_theta,
    float r_max,
    ChainResult *d_results // output, one per chain
);