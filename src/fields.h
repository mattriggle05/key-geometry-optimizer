#pragma once

// a single finger's effort field on a polar grid
struct FingerField
{
    float *data; // flat array [n_r * n_theta]
    float rest_x;
    float rest_y;
    int n_r;
    int n_theta;
    float r_max;
};

// allocates and fills a dummy field with a simple radial gradient
// in the real project this loads from a file
FingerField make_dummy_field(float rest_x, float rest_y,
                             int n_r, int n_theta, float r_max);

void free_field(FingerField &f);