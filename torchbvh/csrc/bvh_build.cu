#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cstdint>
#include <climits>
#include <tuple>

#include "implicit_tree.cuh"
#include "morton.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>


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

// Each block owns one contiguous Morton subtree.  It publishes all real leaves
// and up to seven lower levels without global atomics; only the completed root
// participates in the upper-tree atomic climb.
template <int D, int MAX_GROUP = 128>
__global__ void build_bvh_lower_cooperative_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ sorted_indices,
    float* __restrict__ node_aabbs,
    int B,
    int N,
    int nr,
    int leaf_level,
    int subtree_levels
) {
    const int group = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= B) return;

    namespace tree = implicit_bvh::tree;
    const int group_size = 1 << subtree_levels;
    const int leaf_pos = group * group_size + tid;
    const int local_leaf = group_size - 1 + tid;
    __shared__ float shared_aabbs[(2 * MAX_GROUP - 1) * 2 * D];
    float* sample_aabbs = node_aabbs + static_cast<int64_t>(b) * nr * 2 * D;

    if (tid < group_size) {
        float* dst = shared_aabbs + local_leaf * 2 * D;
        if (leaf_pos < N) {
            const int original_idx = static_cast<int>(
                sorted_indices[static_cast<int64_t>(b) * N + leaf_pos]);
            const float* point = points +
                (static_cast<int64_t>(b) * N + original_idx) * D;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dst[d] = point[d];
                dst[D + d] = point[d];
            }
            const int implicit_idx = tree::first_index_at_level(leaf_level) + leaf_pos;
            store_aabb<D>(sample_aabbs, tree::memory_index(N, implicit_idx), dst);
        } else {
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dst[d] = 1.0e30f;
                dst[D + d] = -1.0e30f;
            }
        }
    }
    __syncthreads();

    for (int step = 1; step <= subtree_levels; ++step) {
        const int width = group_size >> step;
        if (tid < width) {
            const int level = leaf_level - step;
            const int implicit_idx = tree::first_index_at_level(level) + group * width + tid;
            if (!tree::is_virtual(N, implicit_idx)) {
                const int local_node = width - 1 + tid;
                const int local_left = 2 * local_node + 1;
                const int local_right = local_left + 1;
                float* dst = shared_aabbs + local_node * 2 * D;
                const float* left = shared_aabbs + local_left * 2 * D;
                const int right_idx = tree::right_child(implicit_idx);
                const bool right_real = !tree::is_virtual(N, right_idx);
                const float* right = shared_aabbs + local_right * 2 * D;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    dst[d] = right_real ? fminf(left[d], right[d]) : left[d];
                    dst[D + d] = right_real
                        ? fmaxf(left[D + d], right[D + d])
                        : left[D + d];
                }
                store_aabb<D>(sample_aabbs, tree::memory_index(N, implicit_idx), dst);
            }
        }
        __syncthreads();
    }
}

template <int D>
__global__ void build_bvh_upper_from_subtrees_kernel(
    float* __restrict__ node_aabbs,
    int* __restrict__ parent_counters,
    int B,
    int N,
    int nr,
    int cut_level,
    int subtree_roots
) {
    const int root_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= B || root_pos >= subtree_roots) return;

    namespace tree = implicit_bvh::tree;
    int implicit_idx = tree::first_index_at_level(cut_level) + root_pos;
    float* sample_aabbs = node_aabbs + static_cast<int64_t>(b) * nr * 2 * D;
    int* sample_counters = parent_counters + static_cast<int64_t>(b) * nr;
    float current[2 * D];

    while (implicit_idx > 0) {
        const int parent_idx = tree::parent(implicit_idx);
        const int left_idx = tree::left_child(parent_idx);
        const int right_idx = tree::right_child(parent_idx);
        const bool right_real = !tree::is_virtual(N, right_idx);
        const int parent_mem = tree::memory_index(N, parent_idx);
        if (!right_real) {
            copy_aabb<D>(sample_aabbs, current, tree::memory_index(N, left_idx));
            store_aabb<D>(sample_aabbs, parent_mem, current);
            __threadfence();
            implicit_idx = parent_idx;
            continue;
        }
        if (atomicAdd(sample_counters + parent_mem, 1) == 0) return;
        merge_aabbs<D>(
            sample_aabbs,
            tree::memory_index(N, left_idx),
            tree::memory_index(N, right_idx),
            current);
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

template <int D, int THREADS>
__global__ void point_bounds_batched_kernel(
    const float* __restrict__ points,
    float* __restrict__ scene_min,
    float* __restrict__ scene_max,
    int N
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float mins[THREADS * D];
    __shared__ float maxs[THREADS * D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        float lo = 1.0e30f;
        float hi = -1.0e30f;
        for (int n = tid; n < N; n += THREADS) {
            const float value = points[(static_cast<int64_t>(b) * N + n) * D + d];
            lo = fminf(lo, value);
            hi = fmaxf(hi, value);
        }
        mins[tid * D + d] = lo;
        maxs[tid * D + d] = hi;
    }
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                mins[tid * D + d] = fminf(mins[tid * D + d], mins[(tid + stride) * D + d]);
                maxs[tid * D + d] = fmaxf(maxs[tid * D + d], maxs[(tid + stride) * D + d]);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            scene_min[b * D + d] = mins[d];
            scene_max[b * D + d] = maxs[d];
        }
    }
}

template <int D>
__global__ void morton_codes_u32_batched_kernel(
    const float* __restrict__ points,
    uint32_t* __restrict__ codes,
    int B,
    int N,
    const float* __restrict__ scene_min,
    const float* __restrict__ scene_max
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(B) * N) return;
    const int b = static_cast<int>(idx / N);
    const float* point = points + idx * D;
    const float* lo = scene_min + b * D;
    const float* hi = scene_max + b * D;
    if constexpr (D == 2) {
        codes[idx] = implicit_bvh::morton::morton_encode_2d(
            point[0], point[1], make_float2(lo[0], lo[1]), make_float2(hi[0], hi[1]));
    } else {
        codes[idx] = implicit_bvh::morton::morton_encode_3d(
            point[0], point[1], point[2],
            make_float3(lo[0], lo[1], lo[2]), make_float3(hi[0], hi[1], hi[2]));
    }
}

__global__ void segmented_iota_i32_kernel(int* values, int total, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) values[idx] = idx % N;
}

__global__ void segment_offsets_kernel(int* offsets, int B, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= B) offsets[idx] = idx * N;
}

__global__ void i32_to_i64_kernel(const int* input, int64_t* output, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) output[idx] = static_cast<int64_t>(input[idx]);
}

template <int D>
std::tuple<torch::Tensor, torch::Tensor> compute_point_bounds_batched_narrow(
    torch::Tensor points
) {
    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    auto scene_min = torch::empty({B, D}, points.options());
    auto scene_max = torch::empty({B, D}, points.options());
    constexpr int threads = 256;
    point_bounds_batched_kernel<D, threads><<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(), scene_min.data_ptr<float>(), scene_max.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scene_min, scene_max};
}

template <int D>
torch::Tensor narrow_morton_sort_batched(
    torch::Tensor points,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int total = B * N;
    auto i32_options = points.options().dtype(torch::kInt32);
    auto i64_options = points.options().dtype(torch::kInt64);
    auto byte_options = points.options().dtype(torch::kByte);
    auto codes_in = torch::empty({total}, i32_options);
    auto codes_out = torch::empty({total}, i32_options);
    auto values_in = torch::empty({total}, i32_options);
    auto values_out = torch::empty({total}, i32_options);
    auto offsets = torch::empty({B + 1}, i32_options);
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    morton_codes_u32_batched_kernel<D><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), reinterpret_cast<uint32_t*>(codes_in.data_ptr<int>()),
        B, N, scene_min.data_ptr<float>(), scene_max.data_ptr<float>());
    segmented_iota_i32_kernel<<<blocks, threads, 0, stream>>>(values_in.data_ptr<int>(), total, N);
    segment_offsets_kernel<<<(B + 1 + threads - 1) / threads, threads, 0, stream>>>(
        offsets.data_ptr<int>(), B, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t temp_bytes = 0;
    constexpr int end_bit = D == 3 ? 30 : 32;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes,
        reinterpret_cast<uint32_t*>(codes_in.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(codes_out.data_ptr<int>()),
        values_in.data_ptr<int>(), values_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream);
    auto temp = torch::empty({static_cast<int64_t>(temp_bytes + 1)}, byte_options);
    cub::DeviceSegmentedRadixSort::SortPairs(
        temp.data_ptr(), temp_bytes,
        reinterpret_cast<uint32_t*>(codes_in.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(codes_out.data_ptr<int>()),
        values_in.data_ptr<int>(), values_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream);
    auto sorted_indices = torch::empty({total}, i64_options);
    i32_to_i64_kernel<<<blocks, threads, 0, stream>>>(
        values_out.data_ptr<int>(), sorted_indices.data_ptr<int64_t>(), total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sorted_indices.view({B, N});
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

template <int D>
torch::Tensor build_node_aabbs_batched_cooperative(
    torch::Tensor points,
    torch::Tensor sorted_indices,
    int B,
    int N,
    int nr,
    int leaf_level
) {
    auto node_aabbs = torch::empty({B, nr, 2 * D}, points.options());
    constexpr int max_subtree_levels = 8;
    const int subtree_levels = leaf_level < max_subtree_levels
        ? leaf_level : max_subtree_levels;
    const int subtree_roots = implicit_bvh::tree::real_nodes_at_level(
        N, leaf_level - subtree_levels);
    const dim3 lower_blocks(subtree_roots, B);
    if (subtree_levels <= 7) {
        build_bvh_lower_cooperative_kernel<D, 128><<<
            lower_blocks, 128, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            points.data_ptr<float>(), sorted_indices.data_ptr<int64_t>(),
            node_aabbs.data_ptr<float>(), B, N, nr, leaf_level, subtree_levels);
    } else {
        build_bvh_lower_cooperative_kernel<D, 256><<<
            lower_blocks, 256, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            points.data_ptr<float>(), sorted_indices.data_ptr<int64_t>(),
            node_aabbs.data_ptr<float>(), B, N, nr, leaf_level, subtree_levels);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int cut_level = leaf_level - subtree_levels;
    if (cut_level > 0) {
        auto counters = torch::zeros({B, nr}, points.options().dtype(torch::kInt32));
        constexpr int threads = 128;
        const dim3 upper_blocks((subtree_roots + threads - 1) / threads, B);
        build_bvh_upper_from_subtrees_kernel<D><<<
            upper_blocks, threads, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            node_aabbs.data_ptr<float>(), counters.data_ptr<int>(), B, N, nr,
            cut_level, subtree_roots);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return node_aabbs;
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int, int, int, int,
    torch::Tensor, torch::Tensor, torch::Tensor
> build_bvh_batched_cooperative_cuda(torch::Tensor points) {
    const char* op = "build_bvh_batched_cooperative";
    TORCH_CHECK(points.is_cuda() && points.is_contiguous()
                && points.scalar_type() == torch::kFloat32,
                op, ": points must be contiguous CUDA float32");
    TORCH_CHECK(points.dim() == 3 && points.size(0) >= 1 && points.size(1) >= 1
                && (points.size(2) == 2 || points.size(2) == 3),
                op, ": points must have shape (B, N, 2|3)");
    TORCH_CHECK(points.size(0) <= INT_MAX && points.size(1) <= INT_MAX
                && points.size(0) * points.size(1) <= INT_MAX,
                op, ": total point count must fit in signed int32");
    c10::cuda::CUDAGuard device_guard(points.device());
    namespace tree = implicit_bvh::tree;
    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int dim = static_cast<int>(points.size(2));
    const int leaf = tree::leaf_level(N);
    const int lv = tree::virtual_leaves(N);
    const int nr = tree::real_node_count(N);
    torch::Tensor scene_min, scene_max, sorted_indices, node_aabbs;
    if (dim == 2) {
        std::tie(scene_min, scene_max) = compute_point_bounds_batched_narrow<2>(points);
        sorted_indices = narrow_morton_sort_batched<2>(points, scene_min, scene_max);
        node_aabbs = build_node_aabbs_batched_cooperative<2>(
            points, sorted_indices, B, N, nr, leaf);
    } else {
        std::tie(scene_min, scene_max) = compute_point_bounds_batched_narrow<3>(points);
        sorted_indices = narrow_morton_sort_batched<3>(points, scene_min, scene_max);
        node_aabbs = build_node_aabbs_batched_cooperative<3>(
            points, sorted_indices, B, N, nr, leaf);
    }
    const auto traversal_options = points.options().dtype(torch::kInt32);
    auto [left_child_mem, right_child_mem, mem_to_leaf] =
        build_traversal_data(traversal_options, N, nr, leaf);
    return std::make_tuple(
        node_aabbs, sorted_indices, scene_min, scene_max,
        B, N, nr, leaf, lv, dim,
        left_child_mem, right_child_mem, mem_to_leaf);
}

torch::Tensor morton_sort_points_batched_cuda(torch::Tensor points) {
    const char* op = "morton_sort_points_batched";
    TORCH_CHECK(points.is_cuda() && points.is_contiguous()
                && points.scalar_type() == torch::kFloat32,
                op, ": points must be contiguous CUDA float32");
    TORCH_CHECK(points.dim() == 3 && points.size(0) >= 1 && points.size(1) >= 1
                && (points.size(2) == 2 || points.size(2) == 3),
                op, ": points must have shape (B, N, 2|3)");
    c10::cuda::CUDAGuard device_guard(points.device());
    auto scene_min = std::get<0>(points.min(1)).contiguous();
    auto scene_max = std::get<0>(points.max(1)).contiguous();
    torch::Tensor codes;
    if (points.size(2) == 2) {
        codes = compute_morton_codes_batched<2>(points, scene_min, scene_max);
    } else {
        codes = compute_morton_codes_batched<3>(points, scene_min, scene_max);
    }
    return std::get<1>(codes.sort(1, false)).contiguous();
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
