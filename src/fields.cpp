#include "fields.h"
#include <cmath>
#include <cstdlib>

FingerField make_dummy_field(float rest_x, float rest_y,
                             int n_r, int n_theta, float r_max)
{
    FingerField f;
    f.rest_x = rest_x;
    f.rest_y = rest_y;
    f.n_r = n_r;
    f.n_theta = n_theta;
    f.r_max = r_max;
    f.data = new float[n_r * n_theta];

    // simple radial gradient: effort increases with distance
    // theta bias: higher effort on the right side (positive x)
    // this mimics a finger that resists lateral movement
    for (int r = 0; r < n_r; r++)
    {
        for (int t = 0; t < n_theta; t++)
        {
            float r_norm = (float)r / (n_r - 1); // 0 to 1
            float theta = (float)t / n_theta * 6.2832f;
            float dir_bias = 0.5f + 0.5f * cosf(theta); // 1.0 ahead, 0.0 behind
            f.data[r * n_theta + t] = r_norm * (0.5f + 0.5f * dir_bias);
        }
    }

    return f;
}

void free_field(FingerField &f)
{
    delete[] f.data;
    f.data = nullptr;
}