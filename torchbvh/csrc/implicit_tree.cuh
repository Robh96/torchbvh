#pragma once

#include <cstdint>

#if !defined(__CUDA_ARCH__) && defined(_MSC_VER)
#include <intrin.h>
#endif

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace implicit_bvh {
namespace tree {

struct Range {
    int first;
    int last;
};

__host__ __device__ inline int popcount_u32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(x));
#else
    return __builtin_popcount(x);
#endif
#endif
}

__host__ __device__ inline int floor_log2_u32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return 31 - __clz(x);
#else
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse(&index, x);
    return static_cast<int>(index);
#else
    return 31 - __builtin_clz(x);
#endif
#endif
}

__host__ __device__ inline int ceil_log2_u32(uint32_t x) {
    return x <= 1 ? 0 : floor_log2_u32(x - 1) + 1;
}

__host__ __device__ inline int virtual_leaves(int t) {
    const int level = ceil_log2_u32(static_cast<uint32_t>(t));
    return (1 << level) - t;
}

__host__ __device__ inline int real_node_count(int t) {
    const int lv = virtual_leaves(t);
    return 2 * t - 1 + popcount_u32(static_cast<uint32_t>(lv));
}

__host__ __device__ inline int leaf_level(int t) {
    return ceil_log2_u32(static_cast<uint32_t>(t));
}

__host__ __device__ inline int virtual_nodes_at_level(int t, int level) {
    const int leaf = leaf_level(t);
    const int lvl = virtual_leaves(t) >> (leaf - level + 1);
    return 2 * lvl - popcount_u32(static_cast<uint32_t>(lvl));
}

__host__ __device__ inline int level_of(int implicit_idx) {
    return floor_log2_u32(static_cast<uint32_t>(implicit_idx + 1));
}

__host__ __device__ inline int first_index_at_level(int level) {
    return (1 << level) - 1;
}

__host__ __device__ inline int nodes_at_level(int level) {
    return 1 << level;
}

__host__ __device__ inline int real_nodes_at_level(int t, int level) {
    const int leaf = leaf_level(t);
    return nodes_at_level(level) - (virtual_leaves(t) >> (leaf - level));
}

__host__ __device__ inline int memory_index(int t, int implicit_idx) {
    const int level = level_of(implicit_idx);
    return implicit_idx - virtual_nodes_at_level(t, level);
}

__host__ __device__ inline bool is_virtual(int t, int implicit_idx) {
    const int level = level_of(implicit_idx);
    const int first = first_index_at_level(level);
    return implicit_idx - first >= real_nodes_at_level(t, level);
}

__host__ __device__ inline Range level_real_range(int t, int level) {
    const int first = first_index_at_level(level);
    const int nreal = real_nodes_at_level(t, level);
    return Range{first, first + nreal - 1};
}

__host__ __device__ inline int left_child(int i) {
    return 2 * i + 1;
}

__host__ __device__ inline int right_child(int i) {
    return 2 * i + 2;
}

__host__ __device__ inline int parent(int i) {
    return (i - 1) >> 1;
}

__host__ __device__ inline int descendant(int i, int n, int k) {
    return ((i + 1) << n) - 1 + k;
}

__host__ __device__ inline int ancestor(int j, int n, int k) {
    return ((j + 1 - k) >> n) - 1;
}

}  // namespace tree
}  // namespace implicit_bvh
