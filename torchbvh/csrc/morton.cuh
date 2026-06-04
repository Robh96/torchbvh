#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace implicit_bvh {
namespace morton {

__host__ __device__ inline float clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

__host__ __device__ inline float normalize_axis(float x, float lo, float hi) {
    const float extent = hi - lo;
    if (extent <= 0.0f) {
        return 0.0f;
    }
    return clamp01((x - lo) / extent);
}

__host__ __device__ inline uint32_t quantize_normalized(float x, uint32_t bins) {
    const float scaled = clamp01(x) * static_cast<float>(bins);
    const uint32_t q = static_cast<uint32_t>(scaled);
    return q >= bins ? bins - 1 : q;
}

__host__ __device__ inline uint32_t morton_split3(uint32_t x) {
    x &= 0x000003ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8)) & 0x0300f00fu;
    x = (x | (x << 4)) & 0x030c30c3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

__host__ __device__ inline uint32_t morton_encode_3d(
    float x, float y, float z, float3 scene_min, float3 scene_max) {
    const uint32_t xi = quantize_normalized(normalize_axis(x, scene_min.x, scene_max.x), 1024u);
    const uint32_t yi = quantize_normalized(normalize_axis(y, scene_min.y, scene_max.y), 1024u);
    const uint32_t zi = quantize_normalized(normalize_axis(z, scene_min.z, scene_max.z), 1024u);
    return morton_split3(xi) | (morton_split3(yi) << 1) | (morton_split3(zi) << 2);
}

__host__ __device__ inline uint32_t morton_split2(uint32_t x) {
    x &= 0x0000ffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

__host__ __device__ inline uint32_t morton_encode_2d(
    float x, float y, float2 scene_min, float2 scene_max) {
    const uint32_t xi = quantize_normalized(normalize_axis(x, scene_min.x, scene_max.x), 65536u);
    const uint32_t yi = quantize_normalized(normalize_axis(y, scene_min.y, scene_max.y), 65536u);
    return morton_split2(xi) | (morton_split2(yi) << 1);
}

}  // namespace morton
}  // namespace implicit_bvh
