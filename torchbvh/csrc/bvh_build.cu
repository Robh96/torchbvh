#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cstdint>
#include <tuple>

#include "implicit_tree.cuh"
#include "morton.cuh"


template <int D>
__global__ void morton_codes_kernel(
    const float* __restrict__ points,
    int64_t* __restrict__ codes,
    int64_t n,
    const float* __restrict__ scene_min,
    const float* __restrict__ scene_max
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    if constexpr (D == 2) {
        const float2 lo = make_float2(scene_min[0], scene_min[1]);
        const float2 hi = make_float2(scene_max[0], scene_max[1]);
        codes[idx] = static_cast<int64_t>(implicit_bvh::morton::morton_encode_2d(
            points[idx * D + 0],
            points[idx * D + 1],
            lo,
            hi
        ));
    } else {
        const float3 lo = make_float3(scene_min[0], scene_min[1], scene_min[2]);
        const float3 hi = make_float3(scene_max[0], scene_max[1], scene_max[2]);
        codes[idx] = static_cast<int64_t>(implicit_bvh::morton::morton_encode_3d(
            points[idx * D + 0],
            points[idx * D + 1],
            points[idx * D + 2],
            lo,
            hi
        ));
    }
}

template <int D>
__global__ void morton_codes_batched_kernel(
    const float* __restrict__ points,
    int64_t* __restrict__ codes,
    int B,
    int N,
    const float* __restrict__ scene_min,
    const float* __restrict__ scene_max
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(B) * N;
    if (idx >= total) {
        return;
    }

    const int b = static_cast<int>(idx / N);
    const int n = static_cast<int>(idx - static_cast<int64_t>(b) * N);
    const float* point = points + (static_cast<int64_t>(b) * N + n) * D;
    const float* lo_ptr = scene_min + b * D;
    const float* hi_ptr = scene_max + b * D;

    if constexpr (D == 2) {
        const float2 lo = make_float2(lo_ptr[0], lo_ptr[1]);
        const float2 hi = make_float2(hi_ptr[0], hi_ptr[1]);
        codes[idx] = static_cast<int64_t>(implicit_bvh::morton::morton_encode_2d(
            point[0],
            point[1],
            lo,
            hi
        ));
    } else {
        const float3 lo = make_float3(lo_ptr[0], lo_ptr[1], lo_ptr[2]);
        const float3 hi = make_float3(hi_ptr[0], hi_ptr[1], hi_ptr[2]);
        codes[idx] = static_cast<int64_t>(implicit_bvh::morton::morton_encode_3d(
            point[0],
            point[1],
            point[2],
            lo,
            hi
        ));
    }
}

template <int D, int V>
__global__ void primitive_bounds_batched_kernel(
    const float* __restrict__ primitives,
    float* __restrict__ centers,
    float* __restrict__ leaf_aabbs,
    int64_t total_primitives
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_primitives) {
        return;
    }

    const float* primitive = primitives + idx * V * D;
    float* center = centers + idx * D;
    float* aabb = leaf_aabbs + idx * 2 * D;

    #pragma unroll
    for (int d = 0; d < D; ++d) {
        float lo = primitive[d];
        float hi = primitive[d];
        float sum = primitive[d];
        #pragma unroll
        for (int vertex = 1; vertex < V; ++vertex) {
            const float value = primitive[vertex * D + d];
            lo = fminf(lo, value);
            hi = fmaxf(hi, value);
            sum += value;
        }
        center[d] = sum / static_cast<float>(V);
        aabb[d] = lo;
        aabb[D + d] = hi;
    }
}

template <int D, int V>
std::tuple<torch::Tensor, torch::Tensor> primitive_bounds_batched(torch::Tensor primitives) {
    const int64_t B = primitives.size(0);
    const int64_t N = primitives.size(1);
    auto centers = torch::empty({B, N, D}, primitives.options());
    auto leaf_aabbs = torch::empty({B, N, 2 * D}, primitives.options());
    const int64_t total = B * N;

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    primitive_bounds_batched_kernel<D, V><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        primitives.data_ptr<float>(),
        centers.data_ptr<float>(),
        leaf_aabbs.data_ptr<float>(),
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(centers, leaf_aabbs);
}

template <int D>
__device__ inline void copy_aabb(
    const float* __restrict__ node_aabbs,
    float* __restrict__ out,
    int src_mem
) {
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        out[d] = node_aabbs[src_mem * 2 * D + d];
        out[D + d] = node_aabbs[src_mem * 2 * D + D + d];
    }
}

template <int D>
__device__ inline void store_aabb(
    float* __restrict__ node_aabbs,
    int dst_mem,
    const float* __restrict__ values
) {
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        node_aabbs[dst_mem * 2 * D + d] = values[d];
        node_aabbs[dst_mem * 2 * D + D + d] = values[D + d];
    }
}

template <int D>
__device__ inline void merge_aabbs(
    const float* __restrict__ node_aabbs,
    int left_mem,
    int right_mem,
    float* __restrict__ out
) {
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float left_lo = node_aabbs[left_mem * 2 * D + d];
        const float right_lo = node_aabbs[right_mem * 2 * D + d];
        const float left_hi = node_aabbs[left_mem * 2 * D + D + d];
        const float right_hi = node_aabbs[right_mem * 2 * D + D + d];
        out[d] = fminf(left_lo, right_lo);
        out[D + d] = fmaxf(left_hi, right_hi);
    }
}

template <int D>
__global__ void build_bvh_kernel(
    const float* __restrict__ points,
    const float* __restrict__ leaf_aabbs,
    const int64_t* __restrict__ sorted_indices,
    float* __restrict__ node_aabbs,
    int* __restrict__ parent_counters,
    int t,
    int leaf_level
) {
    const int leaf_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_pos >= t) {
        return;
    }

    namespace tree = implicit_bvh::tree;

    const int original_idx = static_cast<int>(sorted_indices[leaf_pos]);
    int implicit_idx = tree::first_index_at_level(leaf_level) + leaf_pos;
    int mem_idx = tree::memory_index(t, implicit_idx);

    float current[2 * D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        if (leaf_aabbs != nullptr) {
            current[d] = leaf_aabbs[original_idx * 2 * D + d];
            current[D + d] = leaf_aabbs[original_idx * 2 * D + D + d];
        } else {
            const float value = points[original_idx * D + d];
            current[d] = value;
            current[D + d] = value;
        }
    }
    store_aabb<D>(node_aabbs, mem_idx, current);
    __threadfence();

    while (implicit_idx > 0) {
        const int parent_idx = tree::parent(implicit_idx);
        const int left_idx = tree::left_child(parent_idx);
        const int right_idx = tree::right_child(parent_idx);
        const bool right_real = !tree::is_virtual(t, right_idx);
        const int parent_mem = tree::memory_index(t, parent_idx);

        if (!right_real) {
            const int left_mem = tree::memory_index(t, left_idx);
            copy_aabb<D>(node_aabbs, current, left_mem);
            store_aabb<D>(node_aabbs, parent_mem, current);
            __threadfence();
            implicit_idx = parent_idx;
            continue;
        }

        const int previous = atomicAdd(parent_counters + parent_mem, 1);
        if (previous == 0) {
            return;
        }

        const int left_mem = tree::memory_index(t, left_idx);
        const int right_mem = tree::memory_index(t, right_idx);
        merge_aabbs<D>(node_aabbs, left_mem, right_mem, current);
        store_aabb<D>(node_aabbs, parent_mem, current);
        __threadfence();
        implicit_idx = parent_idx;
    }
}

template <int D>
__global__ void build_bvh_batched_kernel(
    const float* __restrict__ points,
    const float* __restrict__ leaf_aabbs,
    const int64_t* __restrict__ sorted_indices,
    float* __restrict__ node_aabbs,
    int* __restrict__ parent_counters,
    int B,
    int N,
    int nr,
    int leaf_level
) {
    const int leaf_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || leaf_pos >= N) {
        return;
    }

    namespace tree = implicit_bvh::tree;

    const int64_t batch_points_offset = static_cast<int64_t>(b) * N * D;
    const int64_t batch_aabb_offset = static_cast<int64_t>(b) * nr * 2 * D;
    const int64_t batch_sorted_offset = static_cast<int64_t>(b) * N;
    const int64_t batch_counter_offset = static_cast<int64_t>(b) * nr;

    const int original_idx = static_cast<int>(sorted_indices[batch_sorted_offset + leaf_pos]);
    int implicit_idx = tree::first_index_at_level(leaf_level) + leaf_pos;
    int mem_idx = tree::memory_index(N, implicit_idx);

    float* sample_aabbs = node_aabbs + batch_aabb_offset;
    int* sample_counters = parent_counters + batch_counter_offset;

    float current[2 * D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        if (leaf_aabbs != nullptr) {
            const int64_t offset = (static_cast<int64_t>(b) * N + original_idx) * 2 * D;
            current[d] = leaf_aabbs[offset + d];
            current[D + d] = leaf_aabbs[offset + D + d];
        } else {
            const float value = points[batch_points_offset + original_idx * D + d];
            current[d] = value;
            current[D + d] = value;
        }
    }
    store_aabb<D>(sample_aabbs, mem_idx, current);
    __threadfence();

    while (implicit_idx > 0) {
        const int parent_idx = tree::parent(implicit_idx);
        const int left_idx = tree::left_child(parent_idx);
        const int right_idx = tree::right_child(parent_idx);
        const bool right_real = !tree::is_virtual(N, right_idx);
        const int parent_mem = tree::memory_index(N, parent_idx);

        if (!right_real) {
            const int left_mem = tree::memory_index(N, left_idx);
            copy_aabb<D>(sample_aabbs, current, left_mem);
            store_aabb<D>(sample_aabbs, parent_mem, current);
            __threadfence();
            implicit_idx = parent_idx;
            continue;
        }

        const int previous = atomicAdd(sample_counters + parent_mem, 1);
        if (previous == 0) {
            return;
        }

        const int left_mem = tree::memory_index(N, left_idx);
        const int right_mem = tree::memory_index(N, right_idx);
        merge_aabbs<D>(sample_aabbs, left_mem, right_mem, current);
        store_aabb<D>(sample_aabbs, parent_mem, current);
        __threadfence();
        implicit_idx = parent_idx;
    }
}

__global__ void build_traversal_arrays_kernel(
    int* __restrict__ left_child_mem_arr,
    int* __restrict__ right_child_mem_arr,
    int* __restrict__ mem_to_leaf_arr,
    int t,
    int max_implicit,
    int leaf_level
) {
    const int implicit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (implicit_idx >= max_implicit) {
        return;
    }

    namespace tree = implicit_bvh::tree;
    if (tree::is_virtual(t, implicit_idx)) {
        return;
    }

    const int mem_idx = tree::memory_index(t, implicit_idx);
    const bool is_leaf_node = (tree::level_of(implicit_idx) == leaf_level);

    left_child_mem_arr[mem_idx] = is_leaf_node
        ? -1
        : tree::memory_index(t, tree::left_child(implicit_idx));

    if (is_leaf_node) {
        right_child_mem_arr[mem_idx] = -1;
    } else {
        const int right_idx = tree::right_child(implicit_idx);
        right_child_mem_arr[mem_idx] = tree::is_virtual(t, right_idx)
            ? -1
            : tree::memory_index(t, right_idx);
    }

    mem_to_leaf_arr[mem_idx] = is_leaf_node
        ? (implicit_idx - tree::first_index_at_level(leaf_level))
        : -1;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_traversal_data(
    const at::TensorOptions& int_options,
    int t,
    int nr,
    int leaf_level
) {
    auto left_child_mem = torch::empty({nr}, int_options.dtype(torch::kInt32));
    auto right_child_mem = torch::empty({nr}, int_options.dtype(torch::kInt32));
    auto mem_to_leaf = torch::empty({nr}, int_options.dtype(torch::kInt32));

    const int max_implicit = (1 << (leaf_level + 1)) - 1;
    constexpr int threads = 256;
    const int blocks = (max_implicit + threads - 1) / threads;
    build_traversal_arrays_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        left_child_mem.data_ptr<int>(),
        right_child_mem.data_ptr<int>(),
        mem_to_leaf.data_ptr<int>(),
        t,
        max_implicit,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(left_child_mem, right_child_mem, mem_to_leaf);
}

template <int D>
torch::Tensor compute_morton_codes(torch::Tensor points, torch::Tensor scene_min, torch::Tensor scene_max) {
    auto codes = torch::empty({points.size(0)}, points.options().dtype(torch::kInt64));
    const int64_t n = points.size(0);
    if (n == 0) {
        return codes;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    morton_codes_kernel<D><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        n,
        scene_min.data_ptr<float>(),
        scene_max.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return codes;
}

template <int D>
torch::Tensor compute_morton_codes_batched(
    torch::Tensor points,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    const int64_t B = points.size(0);
    const int64_t N = points.size(1);
    auto codes = torch::empty({B, N}, points.options().dtype(torch::kInt64));
    const int64_t total = B * N;
    if (total == 0) {
        return codes;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    morton_codes_batched_kernel<D><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        static_cast<int>(B),
        static_cast<int>(N),
        scene_min.data_ptr<float>(),
        scene_max.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return codes;
}

template <int D>
torch::Tensor build_node_aabbs(
    torch::Tensor points,
    torch::Tensor sorted_indices,
    int t,
    int nr,
    int leaf_level,
    const float* leaf_aabbs = nullptr
) {
    auto node_aabbs = torch::empty({nr, 2 * D}, points.options());
    auto counters = torch::zeros({nr}, points.options().dtype(torch::kInt32));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((t + threads - 1) / threads);
    build_bvh_kernel<D><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        leaf_aabbs,
        sorted_indices.data_ptr<int64_t>(),
        node_aabbs.data_ptr<float>(),
        counters.data_ptr<int>(),
        t,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return node_aabbs;
}

template <int D>
torch::Tensor build_node_aabbs_batched(
    torch::Tensor points,
    torch::Tensor sorted_indices,
    int B,
    int N,
    int nr,
    int leaf_level,
    const float* leaf_aabbs = nullptr
) {
    auto node_aabbs = torch::empty({B, nr, 2 * D}, points.options());
    auto counters = torch::zeros({B, nr}, points.options().dtype(torch::kInt32));

    constexpr int threads = 256;
    const int x_blocks = static_cast<int>((N + threads - 1) / threads);
    const dim3 blocks(x_blocks, B);
    build_bvh_batched_kernel<D><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        leaf_aabbs,
        sorted_indices.data_ptr<int64_t>(),
        node_aabbs.data_ptr<float>(),
        counters.data_ptr<int>(),
        B,
        N,
        nr,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return node_aabbs;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int,
    int,
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> build_bvh_cuda(torch::Tensor points) {
    TORCH_CHECK(points.is_cuda(), "build_bvh: points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "build_bvh: points must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "build_bvh: points must be float32");
    TORCH_CHECK(points.dim() == 2, "build_bvh: points must have shape (N, D)");
    TORCH_CHECK(points.size(0) >= 1, "build_bvh: points must contain at least one point");
    TORCH_CHECK(points.size(1) == 2 || points.size(1) == 3, "build_bvh: D must be 2 or 3");

    c10::cuda::CUDAGuard device_guard(points.device());

    namespace tree = implicit_bvh::tree;
    const int t = static_cast<int>(points.size(0));
    const int dim = static_cast<int>(points.size(1));
    const int leaf = tree::leaf_level(t);
    const int lv = tree::virtual_leaves(t);
    const int nr = tree::real_node_count(t);

    const auto min_result = points.min(0);
    const auto max_result = points.max(0);
    torch::Tensor scene_min = std::get<0>(min_result).contiguous();
    torch::Tensor scene_max = std::get<0>(max_result).contiguous();

    torch::Tensor codes;
    torch::Tensor node_aabbs;
    if (dim == 2) {
        codes = compute_morton_codes<2>(points, scene_min, scene_max);
    } else {
        codes = compute_morton_codes<3>(points, scene_min, scene_max);
    }

    const auto sort_result = codes.sort(0, false);
    torch::Tensor sorted_indices = std::get<1>(sort_result).contiguous();

    if (dim == 2) {
        node_aabbs = build_node_aabbs<2>(points, sorted_indices, t, nr, leaf);
    } else {
        node_aabbs = build_node_aabbs<3>(points, sorted_indices, t, nr, leaf);
    }

    const auto skip_opts = points.options().dtype(torch::kInt32);
    auto [left_child_mem, right_child_mem, mem_to_leaf] = build_traversal_data(skip_opts, t, nr, leaf);

    return std::make_tuple(
        node_aabbs, sorted_indices, scene_min, scene_max,
        t, nr, leaf, lv, dim,
        left_child_mem, right_child_mem, mem_to_leaf
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> build_bvh_batched_cuda(torch::Tensor points) {
    TORCH_CHECK(points.is_cuda(), "build_bvh_batched: points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "build_bvh_batched: points must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "build_bvh_batched: points must be float32");
    TORCH_CHECK(points.dim() == 3, "build_bvh_batched: points must have shape (B, N, D)");
    TORCH_CHECK(points.size(0) >= 1, "build_bvh_batched: batch size must be at least 1");
    TORCH_CHECK(points.size(1) >= 1, "build_bvh_batched: points must contain at least one point per sample");
    TORCH_CHECK(points.size(2) == 2 || points.size(2) == 3, "build_bvh_batched: D must be 2 or 3");

    c10::cuda::CUDAGuard device_guard(points.device());

    namespace tree = implicit_bvh::tree;
    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int dim = static_cast<int>(points.size(2));
    const int leaf = tree::leaf_level(N);
    const int lv = tree::virtual_leaves(N);
    const int nr = tree::real_node_count(N);

    const auto min_result = points.min(1);
    const auto max_result = points.max(1);
    torch::Tensor scene_min = std::get<0>(min_result).contiguous();
    torch::Tensor scene_max = std::get<0>(max_result).contiguous();

    torch::Tensor codes;
    torch::Tensor node_aabbs;
    if (dim == 2) {
        codes = compute_morton_codes_batched<2>(points, scene_min, scene_max);
    } else {
        codes = compute_morton_codes_batched<3>(points, scene_min, scene_max);
    }

    const auto sort_result = codes.sort(1, false);
    torch::Tensor sorted_indices = std::get<1>(sort_result).contiguous();

    if (dim == 2) {
        node_aabbs = build_node_aabbs_batched<2>(points, sorted_indices, B, N, nr, leaf);
    } else {
        node_aabbs = build_node_aabbs_batched<3>(points, sorted_indices, B, N, nr, leaf);
    }

    // Traversal arrays are identical for all samples (same N); store one copy of shape (nr,).
    const auto skip_opts = points.options().dtype(torch::kInt32);
    auto [left_child_mem, right_child_mem, mem_to_leaf] = build_traversal_data(skip_opts, N, nr, leaf);

    return std::make_tuple(
        node_aabbs, sorted_indices, scene_min, scene_max,
        B, N, nr, leaf, lv, dim,
        left_child_mem, right_child_mem, mem_to_leaf
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> build_primitive_bvh_batched_cuda(torch::Tensor primitives) {
    TORCH_CHECK(primitives.is_cuda(), "build_primitive_bvh_batched: primitives must be a CUDA tensor");
    TORCH_CHECK(primitives.is_contiguous(), "build_primitive_bvh_batched: primitives must be contiguous");
    TORCH_CHECK(primitives.scalar_type() == torch::kFloat32, "build_primitive_bvh_batched: primitives must be float32");
    TORCH_CHECK(primitives.dim() == 4, "build_primitive_bvh_batched: primitives must have shape (B, F, V, D)");
    TORCH_CHECK(primitives.size(0) >= 1, "build_primitive_bvh_batched: batch size must be at least 1");
    TORCH_CHECK(primitives.size(1) >= 1, "build_primitive_bvh_batched: each sample must contain at least one primitive");

    const int vertices = static_cast<int>(primitives.size(2));
    const int dim = static_cast<int>(primitives.size(3));
    TORCH_CHECK(
        (vertices == 2 && dim == 2) || (vertices == 3 && dim == 3),
        "build_primitive_bvh_batched: expected 2-D segments or 3-D triangles"
    );

    c10::cuda::CUDAGuard device_guard(primitives.device());

    torch::Tensor centers;
    torch::Tensor leaf_aabbs;
    if (dim == 2) {
        std::tie(centers, leaf_aabbs) = primitive_bounds_batched<2, 2>(primitives);
    } else {
        std::tie(centers, leaf_aabbs) = primitive_bounds_batched<3, 3>(primitives);
    }

    namespace tree = implicit_bvh::tree;
    const int B = static_cast<int>(primitives.size(0));
    const int N = static_cast<int>(primitives.size(1));
    const int leaf = tree::leaf_level(N);
    const int lv = tree::virtual_leaves(N);
    const int nr = tree::real_node_count(N);

    const auto lower = leaf_aabbs.slice(2, 0, dim);
    const auto upper = leaf_aabbs.slice(2, dim, 2 * dim);
    torch::Tensor scene_min = std::get<0>(lower.min(1)).contiguous();
    torch::Tensor scene_max = std::get<0>(upper.max(1)).contiguous();

    torch::Tensor codes;
    if (dim == 2) {
        codes = compute_morton_codes_batched<2>(centers, scene_min, scene_max);
    } else {
        codes = compute_morton_codes_batched<3>(centers, scene_min, scene_max);
    }
    const auto sort_result = codes.sort(1, false);
    torch::Tensor sorted_indices = std::get<1>(sort_result).contiguous();

    torch::Tensor node_aabbs;
    if (dim == 2) {
        node_aabbs = build_node_aabbs_batched<2>(
            centers, sorted_indices, B, N, nr, leaf, leaf_aabbs.data_ptr<float>()
        );
    } else {
        node_aabbs = build_node_aabbs_batched<3>(
            centers, sorted_indices, B, N, nr, leaf, leaf_aabbs.data_ptr<float>()
        );
    }

    const auto traversal_options = primitives.options().dtype(torch::kInt32);
    auto [left_child_mem, right_child_mem, mem_to_leaf] = build_traversal_data(
        traversal_options, N, nr, leaf
    );

    return std::make_tuple(
        node_aabbs, sorted_indices, scene_min, scene_max,
        B, N, nr, leaf, lv, dim,
        left_child_mem, right_child_mem, mem_to_leaf
    );
}
