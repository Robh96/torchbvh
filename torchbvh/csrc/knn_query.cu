#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <tuple>

#include "geometry.cuh"
#include "implicit_tree.cuh"


constexpr float kFloatInf = 3.4028234663852886e38F;

template <int K>
__device__ inline void insert_candidate(
    int64_t candidate_idx,
    float candidate_dist,
    int64_t* __restrict__ best_indices,
    float* __restrict__ best_distances
) {
    if (candidate_dist >= best_distances[K - 1]) {
        return;
    }

    best_distances[K - 1] = candidate_dist;
    best_indices[K - 1] = candidate_idx;

    #pragma unroll
    for (int i = K - 1; i >= 1; --i) {
        if (best_distances[i] < best_distances[i - 1]) {
            const float tmp_dist = best_distances[i - 1];
            const int64_t tmp_idx = best_indices[i - 1];
            best_distances[i - 1] = best_distances[i];
            best_indices[i - 1] = best_indices[i];
            best_distances[i] = tmp_dist;
            best_indices[i] = tmp_idx;
        }
    }
}

// Exact closer-child-first traversal shared by the ordinary ordered batched
// kernel and the conditional routed kernel. index_offset maps a branch-local
// primitive index into a concatenated source array.
template <int D, int K>
__device__ inline void traverse_knn_tree(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ q,
    int num_leaves,
    int leaf_level,
    int64_t index_offset,
    int64_t* __restrict__ best_indices,
    float* __restrict__ best_distances
) {
    namespace tree = implicit_bvh::tree;

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_indices[i] = -1;
        best_distances[i] = kFloatInf;
    }

    constexpr int stack_capacity = 64;
    int stack[stack_capacity];
    int stack_size = 0;
    stack[stack_size++] = 0;
    const int first_leaf = tree::first_index_at_level(leaf_level);

    while (stack_size > 0) {
        const int implicit_idx = stack[--stack_size];
        if (tree::is_virtual(num_leaves, implicit_idx)) continue;

        const int mem_idx = tree::memory_index(num_leaves, implicit_idx);
        const float node_dist = min_distance_sq_to_aabb<D>(
            q, node_aabbs + mem_idx * 2 * D);
        if (node_dist > best_distances[K - 1]) continue;

        if (tree::level_of(implicit_idx) == leaf_level) {
            const int leaf_pos = implicit_idx - first_leaf;
            insert_candidate<K>(
                sorted_indices[leaf_pos] + index_offset,
                node_dist,
                best_indices,
                best_distances);
            continue;
        }

        const int left_idx = tree::left_child(implicit_idx);
        const int right_idx = tree::right_child(implicit_idx);
        const bool left_real = !tree::is_virtual(num_leaves, left_idx);
        const bool right_real = !tree::is_virtual(num_leaves, right_idx);
        float left_dist = kFloatInf;
        float right_dist = kFloatInf;
        if (left_real) {
            const int left_mem = tree::memory_index(num_leaves, left_idx);
            left_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + left_mem * 2 * D);
        }
        if (right_real) {
            const int right_mem = tree::memory_index(num_leaves, right_idx);
            right_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + right_mem * 2 * D);
        }

        if (left_dist <= right_dist) {
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity)
                stack[stack_size++] = right_idx;
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity)
                stack[stack_size++] = left_idx;
        } else {
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity)
                stack[stack_size++] = left_idx;
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity)
                stack[stack_size++] = right_idx;
        }
    }
}

template <int D, int K>
__global__ void query_knn_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ query_points,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int num_queries,
    int num_leaves,
    int leaf_level
) {
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) {
        return;
    }

    namespace tree = implicit_bvh::tree;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q[d] = query_points[query_idx * D + d];
    }

    int64_t best_indices[K];
    float best_distances[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_indices[i] = -1;
        best_distances[i] = kFloatInf;
    }

    constexpr int stack_capacity = 64;
    int stack[stack_capacity];
    int stack_size = 0;
    stack[stack_size++] = 0;

    const int first_leaf = tree::first_index_at_level(leaf_level);

    while (stack_size > 0) {
        const int implicit_idx = stack[--stack_size];
        if (tree::is_virtual(num_leaves, implicit_idx)) {
            continue;
        }

        const int mem_idx = tree::memory_index(num_leaves, implicit_idx);
        const float* aabb = node_aabbs + mem_idx * 2 * D;
        const float node_dist = min_distance_sq_to_aabb<D>(q, aabb);
        if (node_dist > best_distances[K - 1]) {
            continue;
        }

        if (tree::level_of(implicit_idx) == leaf_level) {
            const int leaf_pos = implicit_idx - first_leaf;
            const int64_t original_idx = sorted_indices[leaf_pos];
            insert_candidate<K>(original_idx, node_dist, best_indices, best_distances);
            continue;
        }

        const int left_idx = tree::left_child(implicit_idx);
        const int right_idx = tree::right_child(implicit_idx);
        const bool left_real = !tree::is_virtual(num_leaves, left_idx);
        const bool right_real = !tree::is_virtual(num_leaves, right_idx);

        float left_dist = kFloatInf;
        float right_dist = kFloatInf;
        if (left_real) {
            const int left_mem = tree::memory_index(num_leaves, left_idx);
            left_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + left_mem * 2 * D);
        }
        if (right_real) {
            const int right_mem = tree::memory_index(num_leaves, right_idx);
            right_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + right_mem * 2 * D);
        }

        if (left_dist <= right_dist) {
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = right_idx;
            }
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = left_idx;
            }
        } else {
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = left_idx;
            }
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = right_idx;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[query_idx * K + i] = best_indices[i];
        out_distances[query_idx * K + i] = best_distances[i];
    }
}

template <int D, int K>
__global__ void query_knn_ordered_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ query_points,
    const int64_t* __restrict__ query_order,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int num_queries,
    int num_leaves,
    int leaf_level
) {
    const int sorted_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_query_idx >= num_queries) {
        return;
    }

    const int query_idx = static_cast<int>(query_order[sorted_query_idx]);

    namespace tree = implicit_bvh::tree;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q[d] = query_points[query_idx * D + d];
    }

    int64_t best_indices[K];
    float best_distances[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_indices[i] = -1;
        best_distances[i] = kFloatInf;
    }

    constexpr int stack_capacity = 64;
    int stack[stack_capacity];
    int stack_size = 0;
    stack[stack_size++] = 0;

    const int first_leaf = tree::first_index_at_level(leaf_level);

    while (stack_size > 0) {
        const int implicit_idx = stack[--stack_size];
        if (tree::is_virtual(num_leaves, implicit_idx)) {
            continue;
        }

        const int mem_idx = tree::memory_index(num_leaves, implicit_idx);
        const float* aabb = node_aabbs + mem_idx * 2 * D;
        const float node_dist = min_distance_sq_to_aabb<D>(q, aabb);
        if (node_dist > best_distances[K - 1]) {
            continue;
        }

        if (tree::level_of(implicit_idx) == leaf_level) {
            const int leaf_pos = implicit_idx - first_leaf;
            const int64_t original_idx = sorted_indices[leaf_pos];
            insert_candidate<K>(original_idx, node_dist, best_indices, best_distances);
            continue;
        }

        const int left_idx = tree::left_child(implicit_idx);
        const int right_idx = tree::right_child(implicit_idx);
        const bool left_real = !tree::is_virtual(num_leaves, left_idx);
        const bool right_real = !tree::is_virtual(num_leaves, right_idx);

        float left_dist = kFloatInf;
        float right_dist = kFloatInf;
        if (left_real) {
            const int left_mem = tree::memory_index(num_leaves, left_idx);
            left_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + left_mem * 2 * D);
        }
        if (right_real) {
            const int right_mem = tree::memory_index(num_leaves, right_idx);
            right_dist = min_distance_sq_to_aabb<D>(q, node_aabbs + right_mem * 2 * D);
        }

        if (left_dist <= right_dist) {
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = right_idx;
            }
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = left_idx;
            }
        } else {
            if (left_real && left_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = left_idx;
            }
            if (right_real && right_dist <= best_distances[K - 1] && stack_size < stack_capacity) {
                stack[stack_size++] = right_idx;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[query_idx * K + i] = best_indices[i];
        out_distances[query_idx * K + i] = best_distances[i];
    }
}

template <int D, int K>
__global__ void query_knn_batched_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ query_points,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int B,
    int M,
    int num_leaves,
    int num_real_nodes,
    int leaf_level
) {
    const int global_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (global_query_idx >= total_queries) {
        return;
    }

    const int b = global_query_idx / M;
    const int query_idx = global_query_idx - b * M;

    const float* sample_aabbs = node_aabbs + static_cast<int64_t>(b) * num_real_nodes * 2 * D;
    const int64_t* sample_sorted_indices = sorted_indices + static_cast<int64_t>(b) * num_leaves;
    const float* sample_queries = query_points + static_cast<int64_t>(b) * M * D;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q[d] = sample_queries[query_idx * D + d];
    }

    int64_t best_indices[K];
    float best_distances[K];
    traverse_knn_tree<D, K>(
        sample_aabbs, sample_sorted_indices, q,
        num_leaves, leaf_level, 0, best_indices, best_distances);

    const int64_t out_offset = (static_cast<int64_t>(b) * M + query_idx) * K;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[out_offset + i] = best_indices[i];
        out_distances[out_offset + i] = best_distances[i];
    }
}

template <int D, int K>
__global__ void query_knn_routed_batched_ordered_kernel(
    const float* __restrict__ true_node_aabbs,
    const int64_t* __restrict__ true_sorted_indices,
    const float* __restrict__ false_node_aabbs,
    const int64_t* __restrict__ false_sorted_indices,
    const float* __restrict__ query_points,
    const bool* __restrict__ routes,
    const int64_t* __restrict__ query_order,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int B,
    int M,
    int true_num_leaves,
    int true_num_real_nodes,
    int true_leaf_level,
    int false_num_leaves,
    int false_num_real_nodes,
    int false_leaf_level
) {
    const int global_sorted_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (global_sorted_query_idx >= total_queries) return;

    const int b = global_sorted_query_idx / M;
    const int query_idx = static_cast<int>(query_order[global_sorted_query_idx]);
    const int64_t flat_query_idx = static_cast<int64_t>(b) * M + query_idx;
    const bool route = routes[flat_query_idx];
    const int num_leaves = route ? true_num_leaves : false_num_leaves;
    const int num_real_nodes = route ? true_num_real_nodes : false_num_real_nodes;
    const int leaf_level = route ? true_leaf_level : false_leaf_level;
    const float* all_aabbs = route ? true_node_aabbs : false_node_aabbs;
    const int64_t* all_sorted = route ? true_sorted_indices : false_sorted_indices;
    const float* sample_aabbs = all_aabbs + static_cast<int64_t>(b) * num_real_nodes * 2 * D;
    const int64_t* sample_sorted = all_sorted + static_cast<int64_t>(b) * num_leaves;
    const float* query = query_points + flat_query_idx * D;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) q[d] = query[d];
    int64_t best_indices[K];
    float best_distances[K];
    traverse_knn_tree<D, K>(
        sample_aabbs, sample_sorted, q, num_leaves, leaf_level,
        route ? 0 : static_cast<int64_t>(true_num_leaves),
        best_indices, best_distances);

    const int64_t out_offset = flat_query_idx * K;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[out_offset + i] = best_indices[i];
        out_distances[out_offset + i] = best_distances[i];
    }
}

template <int D, int K>
__global__ void query_knn_batched_ordered_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ query_points,
    const int64_t* __restrict__ query_order,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int B,
    int M,
    int num_leaves,
    int num_real_nodes,
    int leaf_level
) {
    const int global_sorted_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (global_sorted_query_idx >= total_queries) {
        return;
    }

    const int b = global_sorted_query_idx / M;
    const int query_idx = static_cast<int>(query_order[global_sorted_query_idx]);
    const float* sample_aabbs = node_aabbs + static_cast<int64_t>(b) * num_real_nodes * 2 * D;
    const int64_t* sample_sorted_indices = sorted_indices + static_cast<int64_t>(b) * num_leaves;
    const float* sample_queries = query_points + static_cast<int64_t>(b) * M * D;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q[d] = sample_queries[query_idx * D + d];
    }

    int64_t best_indices[K];
    float best_distances[K];
    traverse_knn_tree<D, K>(
        sample_aabbs, sample_sorted_indices, q,
        num_leaves, leaf_level, 0, best_indices, best_distances);

    const int64_t out_offset = (static_cast<int64_t>(b) * M + query_idx) * K;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[out_offset + i] = best_indices[i];
        out_distances[out_offset + i] = best_distances[i];
    }
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_impl(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int leaf_level
) {
    const int64_t num_queries = query_points.size(0);
    auto indices = torch::empty({num_queries, K}, sorted_indices.options());
    auto distances = torch::empty({num_queries, K}, query_points.options());

    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_queries + threads - 1) / threads);
    query_knn_kernel<D, K><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_aabbs.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        query_points.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        static_cast<int>(num_queries),
        num_leaves,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(indices, distances);
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_ordered_impl(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int leaf_level
) {
    const int64_t num_queries = query_points.size(0);
    auto indices = torch::empty({num_queries, K}, sorted_indices.options());
    auto distances = torch::empty({num_queries, K}, query_points.options());

    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_queries + threads - 1) / threads);
    query_knn_ordered_kernel<D, K><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_aabbs.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        query_points.data_ptr<float>(),
        query_order.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        static_cast<int>(num_queries),
        num_leaves,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(indices, distances);
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_impl(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int leaf_level
) {
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    auto indices = torch::empty({B, M, K}, sorted_indices.options());
    auto distances = torch::empty({B, M, K}, query_points.options());

    constexpr int threads = 256;
    const int64_t total_queries = B * M;
    const int blocks = static_cast<int>((total_queries + threads - 1) / threads);
    query_knn_batched_kernel<D, K><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_aabbs.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        query_points.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(M),
        num_leaves,
        num_real_nodes,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(indices, distances);
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_ordered_impl(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int leaf_level
) {
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    auto indices = torch::empty({B, M, K}, sorted_indices.options());
    auto distances = torch::empty({B, M, K}, query_points.options());

    constexpr int threads = 256;
    const int64_t total_queries = B * M;
    const int blocks = static_cast<int>((total_queries + threads - 1) / threads);
    query_knn_batched_ordered_kernel<D, K><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_aabbs.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        query_points.data_ptr<float>(),
        query_order.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(M),
        num_leaves,
        num_real_nodes,
        leaf_level
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(indices, distances);
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_routed_batched_ordered_impl(
    torch::Tensor true_node_aabbs,
    torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs,
    torch::Tensor false_sorted_indices,
    torch::Tensor query_points,
    torch::Tensor routes,
    torch::Tensor query_order,
    int true_num_leaves,
    int true_num_real_nodes,
    int true_leaf_level,
    int false_num_leaves,
    int false_num_real_nodes,
    int false_leaf_level
) {
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    auto indices = torch::empty({B, M, K}, true_sorted_indices.options());
    auto distances = torch::empty({B, M, K}, query_points.options());
    constexpr int threads = 256;
    const int64_t total_queries = B * M;
    const int blocks = static_cast<int>((total_queries + threads - 1) / threads);
    query_knn_routed_batched_ordered_kernel<D, K>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            true_node_aabbs.data_ptr<float>(), true_sorted_indices.data_ptr<int64_t>(),
            false_node_aabbs.data_ptr<float>(), false_sorted_indices.data_ptr<int64_t>(),
            query_points.data_ptr<float>(), routes.data_ptr<bool>(), query_order.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>(), distances.data_ptr<float>(),
            static_cast<int>(B), static_cast<int>(M),
            true_num_leaves, true_num_real_nodes, true_leaf_level,
            false_num_leaves, false_num_real_nodes, false_leaf_level);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, distances);
}

template <int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_dispatch_dim(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int leaf_level,
    int dim
) {
    if (dim == 2) {
        return query_knn_impl<2, K>(node_aabbs, sorted_indices, query_points, num_leaves, leaf_level);
    }
    return query_knn_impl<3, K>(node_aabbs, sorted_indices, query_points, num_leaves, leaf_level);
}

template <int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_ordered_dispatch_dim(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int leaf_level,
    int dim
) {
    if (dim == 2) {
        return query_knn_ordered_impl<2, K>(
            node_aabbs, sorted_indices, query_points, query_order, num_leaves, leaf_level);
    }
    return query_knn_ordered_impl<3, K>(
        node_aabbs, sorted_indices, query_points, query_order, num_leaves, leaf_level);
}

template <int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_dispatch_dim(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim
) {
    if (dim == 2) {
        return query_knn_batched_impl<2, K>(
            node_aabbs,
            sorted_indices,
            query_points,
            num_leaves,
            num_real_nodes,
            leaf_level
        );
    }
    return query_knn_batched_impl<3, K>(
        node_aabbs,
        sorted_indices,
        query_points,
        num_leaves,
        num_real_nodes,
        leaf_level
    );
}

template <int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_ordered_dispatch_dim(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim
) {
    if (dim == 2) {
        return query_knn_batched_ordered_impl<2, K>(
            node_aabbs,
            sorted_indices,
            query_points,
            query_order,
            num_leaves,
            num_real_nodes,
            leaf_level
        );
    }
    return query_knn_batched_ordered_impl<3, K>(
        node_aabbs,
        sorted_indices,
        query_points,
        query_order,
        num_leaves,
        num_real_nodes,
        leaf_level
    );
}

template <int K>
std::tuple<torch::Tensor, torch::Tensor> query_knn_routed_batched_ordered_dispatch_dim(
    torch::Tensor true_node_aabbs,
    torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs,
    torch::Tensor false_sorted_indices,
    torch::Tensor query_points,
    torch::Tensor routes,
    torch::Tensor query_order,
    int true_num_leaves,
    int true_num_real_nodes,
    int true_leaf_level,
    int false_num_leaves,
    int false_num_real_nodes,
    int false_leaf_level,
    int dim
) {
    if (dim == 2) {
        return query_knn_routed_batched_ordered_impl<2, K>(
            true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices,
            query_points, routes, query_order,
            true_num_leaves, true_num_real_nodes, true_leaf_level,
            false_num_leaves, false_num_real_nodes, false_leaf_level);
    }
    return query_knn_routed_batched_ordered_impl<3, K>(
        true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices,
        query_points, routes, query_order,
        true_num_leaves, true_num_real_nodes, true_leaf_level,
        false_num_leaves, false_num_real_nodes, false_leaf_level);
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
) {
    TORCH_CHECK(node_aabbs.is_cuda(), "query_knn: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "query_knn: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(query_points.is_cuda(), "query_knn: query_points must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_contiguous(), "query_knn: node_aabbs must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "query_knn: sorted_indices must be contiguous");
    TORCH_CHECK(query_points.is_contiguous(), "query_knn: query_points must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "query_knn: node_aabbs must be float32");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_knn: query_points must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "query_knn: sorted_indices must be int64");
    TORCH_CHECK(query_points.dim() == 2, "query_knn: query_points must have shape (N_queries, D)");
    TORCH_CHECK(dim == 2 || dim == 3, "query_knn: D must be 2 or 3");
    TORCH_CHECK(query_points.size(1) == dim, "query_knn: query_points second dimension must match dim");
    TORCH_CHECK(node_aabbs.dim() == 2, "query_knn: node_aabbs must have shape (num_real_nodes, 2 * D)");
    TORCH_CHECK(node_aabbs.size(1) == 2 * dim, "query_knn: node_aabbs second dimension must be 2 * D");
    TORCH_CHECK(sorted_indices.size(0) == num_leaves, "query_knn: sorted_indices length must match num_leaves");
    TORCH_CHECK(num_leaves >= k, "query_knn: num_leaves must be >= k");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    TORCH_CHECK(node_aabbs.device() == query_points.device(), "query_knn: node_aabbs and query_points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == query_points.device(), "query_knn: sorted_indices and query_points must be on the same device");

    switch (k) {
        case 4:
            return query_knn_dispatch_dim<4>(node_aabbs, sorted_indices, query_points, num_leaves, leaf_level, dim);
        case 8:
            return query_knn_dispatch_dim<8>(node_aabbs, sorted_indices, query_points, num_leaves, leaf_level, dim);
        case 16:
            return query_knn_dispatch_dim<16>(node_aabbs, sorted_indices, query_points, num_leaves, leaf_level, dim);
        default:
            TORCH_CHECK(false, "query_knn: k must be 4, 8, or 16, got ", k);
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_ordered_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
) {
    TORCH_CHECK(node_aabbs.is_cuda(), "query_knn_ordered: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "query_knn_ordered: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(query_points.is_cuda(), "query_knn_ordered: query_points must be a CUDA tensor");
    TORCH_CHECK(query_order.is_cuda(), "query_knn_ordered: query_order must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_contiguous(), "query_knn_ordered: node_aabbs must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "query_knn_ordered: sorted_indices must be contiguous");
    TORCH_CHECK(query_points.is_contiguous(), "query_knn_ordered: query_points must be contiguous");
    TORCH_CHECK(query_order.is_contiguous(), "query_knn_ordered: query_order must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "query_knn_ordered: node_aabbs must be float32");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_knn_ordered: query_points must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "query_knn_ordered: sorted_indices must be int64");
    TORCH_CHECK(query_order.scalar_type() == torch::kInt64, "query_knn_ordered: query_order must be int64");
    TORCH_CHECK(query_points.dim() == 2, "query_knn_ordered: query_points must have shape (N_queries, D)");
    TORCH_CHECK(query_order.dim() == 1, "query_knn_ordered: query_order must have shape (N_queries,)");
    TORCH_CHECK(query_order.size(0) == query_points.size(0), "query_knn_ordered: query_order length must match query count");
    TORCH_CHECK(dim == 2 || dim == 3, "query_knn_ordered: D must be 2 or 3");
    TORCH_CHECK(query_points.size(1) == dim, "query_knn_ordered: query_points second dimension must match dim");
    TORCH_CHECK(node_aabbs.dim() == 2, "query_knn_ordered: node_aabbs must have shape (num_real_nodes, 2 * D)");
    TORCH_CHECK(node_aabbs.size(1) == 2 * dim, "query_knn_ordered: node_aabbs second dimension must be 2 * D");
    TORCH_CHECK(sorted_indices.size(0) == num_leaves, "query_knn_ordered: sorted_indices length must match num_leaves");
    TORCH_CHECK(num_leaves >= k, "query_knn_ordered: num_leaves must be >= k");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    TORCH_CHECK(node_aabbs.device() == query_points.device(), "query_knn_ordered: node_aabbs and query_points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == query_points.device(), "query_knn_ordered: sorted_indices and query_points must be on the same device");
    TORCH_CHECK(query_order.device() == query_points.device(), "query_knn_ordered: query_order and query_points must be on the same device");

    switch (k) {
        case 4:
            return query_knn_ordered_dispatch_dim<4>(
                node_aabbs, sorted_indices, query_points, query_order, num_leaves, leaf_level, dim);
        case 8:
            return query_knn_ordered_dispatch_dim<8>(
                node_aabbs, sorted_indices, query_points, query_order, num_leaves, leaf_level, dim);
        case 16:
            return query_knn_ordered_dispatch_dim<16>(
                node_aabbs, sorted_indices, query_points, query_order, num_leaves, leaf_level, dim);
        default:
            TORCH_CHECK(false, "query_knn_ordered: k must be 4, 8, or 16, got ", k);
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
) {
    TORCH_CHECK(node_aabbs.is_cuda(), "query_knn_batched: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "query_knn_batched: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(query_points.is_cuda(), "query_knn_batched: query_points must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_contiguous(), "query_knn_batched: node_aabbs must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "query_knn_batched: sorted_indices must be contiguous");
    TORCH_CHECK(query_points.is_contiguous(), "query_knn_batched: query_points must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "query_knn_batched: node_aabbs must be float32");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_knn_batched: query_points must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "query_knn_batched: sorted_indices must be int64");
    TORCH_CHECK(query_points.dim() == 3, "query_knn_batched: query_points must have shape (B, M, D)");
    TORCH_CHECK(dim == 2 || dim == 3, "query_knn_batched: D must be 2 or 3");
    TORCH_CHECK(query_points.size(2) == dim, "query_knn_batched: query_points last dimension must match dim");
    TORCH_CHECK(node_aabbs.dim() == 3, "query_knn_batched: node_aabbs must have shape (B, num_real_nodes, 2 * D)");
    TORCH_CHECK(sorted_indices.dim() == 2, "query_knn_batched: sorted_indices must have shape (B, N)");
    TORCH_CHECK(node_aabbs.size(0) == query_points.size(0), "query_knn_batched: node_aabbs batch size must match query_points");
    TORCH_CHECK(sorted_indices.size(0) == query_points.size(0), "query_knn_batched: sorted_indices batch size must match query_points");
    TORCH_CHECK(node_aabbs.size(1) == num_real_nodes, "query_knn_batched: node_aabbs second dimension must match num_real_nodes");
    TORCH_CHECK(node_aabbs.size(2) == 2 * dim, "query_knn_batched: node_aabbs last dimension must be 2 * D");
    TORCH_CHECK(sorted_indices.size(1) == num_leaves, "query_knn_batched: sorted_indices second dimension must match num_leaves");
    TORCH_CHECK(num_leaves >= k, "query_knn_batched: num_leaves must be >= k");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    TORCH_CHECK(node_aabbs.device() == query_points.device(), "query_knn_batched: node_aabbs and query_points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == query_points.device(), "query_knn_batched: sorted_indices and query_points must be on the same device");

    switch (k) {
        case 4:
            return query_knn_batched_dispatch_dim<4>(
                node_aabbs,
                sorted_indices,
                query_points,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        case 8:
            return query_knn_batched_dispatch_dim<8>(
                node_aabbs,
                sorted_indices,
                query_points,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        case 16:
            return query_knn_batched_dispatch_dim<16>(
                node_aabbs,
                sorted_indices,
                query_points,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        default:
            TORCH_CHECK(false, "query_knn_batched: k must be 4, 8, or 16, got ", k);
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_ordered_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
) {
    TORCH_CHECK(node_aabbs.is_cuda(), "query_knn_batched_ordered: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "query_knn_batched_ordered: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(query_points.is_cuda(), "query_knn_batched_ordered: query_points must be a CUDA tensor");
    TORCH_CHECK(query_order.is_cuda(), "query_knn_batched_ordered: query_order must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_contiguous(), "query_knn_batched_ordered: node_aabbs must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "query_knn_batched_ordered: sorted_indices must be contiguous");
    TORCH_CHECK(query_points.is_contiguous(), "query_knn_batched_ordered: query_points must be contiguous");
    TORCH_CHECK(query_order.is_contiguous(), "query_knn_batched_ordered: query_order must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "query_knn_batched_ordered: node_aabbs must be float32");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_knn_batched_ordered: query_points must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "query_knn_batched_ordered: sorted_indices must be int64");
    TORCH_CHECK(query_order.scalar_type() == torch::kInt64, "query_knn_batched_ordered: query_order must be int64");
    TORCH_CHECK(query_points.dim() == 3, "query_knn_batched_ordered: query_points must have shape (B, M, D)");
    TORCH_CHECK(query_order.dim() == 2, "query_knn_batched_ordered: query_order must have shape (B, M)");
    TORCH_CHECK(query_order.size(0) == query_points.size(0), "query_knn_batched_ordered: query_order batch size must match query_points");
    TORCH_CHECK(query_order.size(1) == query_points.size(1), "query_knn_batched_ordered: query_order query count must match query_points");
    TORCH_CHECK(dim == 2 || dim == 3, "query_knn_batched_ordered: D must be 2 or 3");
    TORCH_CHECK(query_points.size(2) == dim, "query_knn_batched_ordered: query_points last dimension must match dim");
    TORCH_CHECK(node_aabbs.dim() == 3, "query_knn_batched_ordered: node_aabbs must have shape (B, num_real_nodes, 2 * D)");
    TORCH_CHECK(sorted_indices.dim() == 2, "query_knn_batched_ordered: sorted_indices must have shape (B, N)");
    TORCH_CHECK(node_aabbs.size(0) == query_points.size(0), "query_knn_batched_ordered: node_aabbs batch size must match query_points");
    TORCH_CHECK(sorted_indices.size(0) == query_points.size(0), "query_knn_batched_ordered: sorted_indices batch size must match query_points");
    TORCH_CHECK(node_aabbs.size(1) == num_real_nodes, "query_knn_batched_ordered: node_aabbs second dimension must match num_real_nodes");
    TORCH_CHECK(node_aabbs.size(2) == 2 * dim, "query_knn_batched_ordered: node_aabbs last dimension must be 2 * D");
    TORCH_CHECK(sorted_indices.size(1) == num_leaves, "query_knn_batched_ordered: sorted_indices second dimension must match num_leaves");
    TORCH_CHECK(num_leaves >= k, "query_knn_batched_ordered: num_leaves must be >= k");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    TORCH_CHECK(node_aabbs.device() == query_points.device(), "query_knn_batched_ordered: node_aabbs and query_points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == query_points.device(), "query_knn_batched_ordered: sorted_indices and query_points must be on the same device");
    TORCH_CHECK(query_order.device() == query_points.device(), "query_knn_batched_ordered: query_order and query_points must be on the same device");

    switch (k) {
        case 4:
            return query_knn_batched_ordered_dispatch_dim<4>(
                node_aabbs,
                sorted_indices,
                query_points,
                query_order,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        case 8:
            return query_knn_batched_ordered_dispatch_dim<8>(
                node_aabbs,
                sorted_indices,
                query_points,
                query_order,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        case 16:
            return query_knn_batched_ordered_dispatch_dim<16>(
                node_aabbs,
                sorted_indices,
                query_points,
                query_order,
                num_leaves,
                num_real_nodes,
                leaf_level,
                dim
            );
        default:
            TORCH_CHECK(false, "query_knn_batched_ordered: k must be 4, 8, or 16, got ", k);
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_routed_batched_ordered_cuda(
    torch::Tensor true_node_aabbs,
    torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs,
    torch::Tensor false_sorted_indices,
    torch::Tensor query_points,
    torch::Tensor routes,
    torch::Tensor query_order,
    int true_num_leaves,
    int true_num_real_nodes,
    int true_leaf_level,
    int false_num_leaves,
    int false_num_real_nodes,
    int false_leaf_level,
    int dim,
    int k
) {
    const char* op = "query_knn_routed_batched_ordered";
    for (const auto& tensor : {true_node_aabbs, false_node_aabbs, query_points}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), op, ": float inputs must be contiguous CUDA tensors");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, op, ": float inputs must be float32");
    }
    for (const auto& tensor : {true_sorted_indices, false_sorted_indices, query_order}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), op, ": index inputs must be contiguous CUDA tensors");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64, op, ": index inputs must be int64");
    }
    TORCH_CHECK(routes.is_cuda() && routes.is_contiguous(), op, ": routes must be a contiguous CUDA tensor");
    TORCH_CHECK(routes.scalar_type() == torch::kBool, op, ": routes must be bool");
    TORCH_CHECK(dim == 2 || dim == 3, op, ": D must be 2 or 3");
    TORCH_CHECK(query_points.dim() == 3 && query_points.size(2) == dim,
                op, ": query_points must have shape (B, M, D)");
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    TORCH_CHECK(routes.dim() == 2 && routes.size(0) == B && routes.size(1) == M,
                op, ": routes must have shape (B, M)");
    TORCH_CHECK(query_order.dim() == 2 && query_order.size(0) == B && query_order.size(1) == M,
                op, ": query_order must have shape (B, M)");
    TORCH_CHECK(true_node_aabbs.dim() == 3 && true_node_aabbs.size(0) == B &&
                true_node_aabbs.size(1) == true_num_real_nodes && true_node_aabbs.size(2) == 2 * dim,
                op, ": true_node_aabbs has incompatible shape");
    TORCH_CHECK(false_node_aabbs.dim() == 3 && false_node_aabbs.size(0) == B &&
                false_node_aabbs.size(1) == false_num_real_nodes && false_node_aabbs.size(2) == 2 * dim,
                op, ": false_node_aabbs has incompatible shape");
    TORCH_CHECK(true_sorted_indices.dim() == 2 && true_sorted_indices.size(0) == B &&
                true_sorted_indices.size(1) == true_num_leaves,
                op, ": true_sorted_indices has incompatible shape");
    TORCH_CHECK(false_sorted_indices.dim() == 2 && false_sorted_indices.size(0) == B &&
                false_sorted_indices.size(1) == false_num_leaves,
                op, ": false_sorted_indices has incompatible shape");
    TORCH_CHECK(true_num_leaves >= k && false_num_leaves >= k,
                op, ": both source counts must be >= k");

    c10::cuda::CUDAGuard device_guard(query_points.device());
    for (const auto& tensor : {true_node_aabbs, true_sorted_indices, false_node_aabbs,
                               false_sorted_indices, routes, query_order}) {
        TORCH_CHECK(tensor.device() == query_points.device(), op, ": all inputs must share a device");
    }

    switch (k) {
        case 4:
            return query_knn_routed_batched_ordered_dispatch_dim<4>(
                true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices,
                query_points, routes, query_order,
                true_num_leaves, true_num_real_nodes, true_leaf_level,
                false_num_leaves, false_num_real_nodes, false_leaf_level, dim);
        case 8:
            return query_knn_routed_batched_ordered_dispatch_dim<8>(
                true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices,
                query_points, routes, query_order,
                true_num_leaves, true_num_real_nodes, true_leaf_level,
                false_num_leaves, false_num_real_nodes, false_leaf_level, dim);
        case 16:
            return query_knn_routed_batched_ordered_dispatch_dim<16>(
                true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices,
                query_points, routes, query_order,
                true_num_leaves, true_num_real_nodes, true_leaf_level,
                false_num_leaves, false_num_real_nodes, false_leaf_level, dim);
        default:
            TORCH_CHECK(false, op, ": k must be 4, 8, or 16, got ", k);
    }
}
