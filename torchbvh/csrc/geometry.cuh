#pragma once

template <int D>
__device__ inline float min_distance_sq_to_aabb(
    const float* __restrict__ q,
    const float* __restrict__ aabb
) {
    float dist = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float lo = aabb[d];
        const float hi = aabb[D + d];
        const float v = q[d];
        const float delta = fmaxf(lo - v, fmaxf(0.0f, v - hi));
        dist += delta * delta;
    }
    return dist;
}

template <int D>
__device__ inline float max_distance_sq_to_aabb(
    const float* __restrict__ q,
    const float* __restrict__ aabb
) {
    float dist = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float lo = aabb[d];
        const float hi = aabb[D + d];
        const float v = q[d];
        const float delta_lo = fabsf(v - lo);
        const float delta_hi = fabsf(v - hi);
        const float delta = fmaxf(delta_lo, delta_hi);
        dist += delta * delta;
    }
    return dist;
}
