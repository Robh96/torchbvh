#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <tuple>

#include "geometry.cuh"
#include "implicit_tree.cuh"


constexpr float kFloatInf = 3.4028234663852886e38F;

__device__ inline unsigned long long pack_cached_stack_entry(int mem_idx, float bound) {
    return (static_cast<unsigned long long>(static_cast<unsigned int>(mem_idx)) << 32) |
        static_cast<unsigned long long>(__float_as_uint(bound));
}

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

template <int D, int K, bool Ordered, bool SpatialOutput = false>
__global__ void query_knn_batched_cached_bounds_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ query_points,
    const int64_t* __restrict__ query_order,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    int B,
    int M,
    int num_leaves,
    int num_real_nodes
) {
    const int launch_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (launch_query_idx >= total_queries) return;

    const int launch_b = launch_query_idx / M;
    const int launch_m = launch_query_idx - launch_b * M;
    const int query_idx = Ordered
        ? static_cast<int>(query_order[static_cast<int64_t>(launch_b) * M + launch_m])
        : launch_m;
    const int64_t flat_query_idx = static_cast<int64_t>(launch_b) * M + query_idx;
    const float* sample_aabbs =
        node_aabbs + static_cast<int64_t>(launch_b) * num_real_nodes * 2 * D;
    const int64_t* sample_sorted =
        sorted_indices + static_cast<int64_t>(launch_b) * num_leaves;

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q[d] = query_points[flat_query_idx * D + d];
    }

    int64_t best_indices[K];
    float best_distances[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_indices[i] = -1;
        best_distances[i] = kFloatInf;
    }

    // A depth-first traversal retains at most one deferred sibling per level.
    // int-indexed trees have at most 31 edges, versus the maintained 64 slots.
    constexpr int stack_capacity = 32;
    unsigned long long stack[stack_capacity];
    int stack_size = 0;
    namespace tree = implicit_bvh::tree;
    const int leaf_level = tree::leaf_level(num_leaves);
    const int first_leaf = tree::first_index_at_level(leaf_level);
    int implicit_idx = 0;
    float node_dist = min_distance_sq_to_aabb<D>(q, sample_aabbs);

    while (true) {
        if (node_dist <= best_distances[K - 1]) {
            if (implicit_idx >= first_leaf) {
                insert_candidate<K>(
                    sample_sorted[implicit_idx - first_leaf],
                    node_dist,
                    best_indices,
                    best_distances);
            } else {
                const int left_idx = tree::left_child(implicit_idx);
                const int right_idx = tree::right_child(implicit_idx);
                const bool left_real = !tree::is_virtual(num_leaves, left_idx);
                const bool right_real = !tree::is_virtual(num_leaves, right_idx);
                const int left_mem = left_real ? tree::memory_index(num_leaves, left_idx) : -1;
                const int right_mem = right_real ? tree::memory_index(num_leaves, right_idx) : -1;
                const float left_dist = left_real
                    ? min_distance_sq_to_aabb<D>(
                          q, sample_aabbs + static_cast<int64_t>(left_mem) * 2 * D)
                    : kFloatInf;
                const float right_dist = right_real
                    ? min_distance_sq_to_aabb<D>(
                          q, sample_aabbs + static_cast<int64_t>(right_mem) * 2 * D)
                    : kFloatInf;
                const bool left_near = left_dist <= right_dist;
                const int near_idx = left_near ? left_idx : right_idx;
                const int far_idx = left_near ? right_idx : left_idx;
                const float near_dist = left_near ? left_dist : right_dist;
                const float far_dist = left_near ? right_dist : left_dist;
                const bool near_real = left_near ? left_real : right_real;
                const bool far_real = left_near ? right_real : left_real;
                const float cutoff = best_distances[K - 1];
                if (far_real && far_dist <= cutoff) {
                    stack[stack_size++] = pack_cached_stack_entry(far_idx, far_dist);
                }
                if (near_real && near_dist <= cutoff) {
                    implicit_idx = near_idx;
                    node_dist = near_dist;
                    continue;
                }
            }
        }
        if (stack_size == 0) break;
        const unsigned long long entry = stack[--stack_size];
        implicit_idx = static_cast<int>(entry >> 32);
        node_dist = __uint_as_float(static_cast<unsigned int>(entry));
    }

    const int64_t out_offset = static_cast<int64_t>(
        SpatialOutput ? launch_query_idx : flat_query_idx) * K;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[out_offset + i] = best_indices[i];
        out_distances[out_offset + i] = best_distances[i];
    }
}

template <int D, int K, bool Ordered, bool SpatialOutput = false>
std::tuple<torch::Tensor, torch::Tensor> launch_query_knn_batched_cached_bounds(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes
) {
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    auto indices = torch::empty({B, M, K}, sorted_indices.options());
    auto distances = torch::empty({B, M, K}, query_points.options());
    constexpr int threads = 256;
    const int64_t total_queries = B * M;
    const int blocks = static_cast<int>((total_queries + threads - 1) / threads);
    query_knn_batched_cached_bounds_kernel<D, K, Ordered, SpatialOutput>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            node_aabbs.data_ptr<float>(), sorted_indices.data_ptr<int64_t>(),
            query_points.data_ptr<float>(),
            Ordered ? query_order.data_ptr<int64_t>() : nullptr,
            indices.data_ptr<int64_t>(), distances.data_ptr<float>(),
            static_cast<int>(B), static_cast<int>(M), num_leaves, num_real_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, distances);
}

// Routed counterpart of the cached-bounds traversal. The output stays in the
// supplied Morton order so the indexed MLS solver can consume it directly.
template <int D, int K>
__global__ void query_knn_routed_batched_cached_bounds_spatial_kernel(
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
    int false_num_leaves,
    int false_num_real_nodes
) {
    const int launch_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (launch_query_idx >= total_queries) return;

    const int b = launch_query_idx / M;
    const int storage_m = launch_query_idx - b * M;
    const int query_idx = static_cast<int>(
        query_order[static_cast<int64_t>(b) * M + storage_m]);
    const int64_t flat_query_idx = static_cast<int64_t>(b) * M + query_idx;
    const bool route = routes[flat_query_idx];
    const int num_leaves = route ? true_num_leaves : false_num_leaves;
    const int num_real_nodes = route ? true_num_real_nodes : false_num_real_nodes;
    const float* all_aabbs = route ? true_node_aabbs : false_node_aabbs;
    const int64_t* all_sorted = route ? true_sorted_indices : false_sorted_indices;
    const float* sample_aabbs =
        all_aabbs + static_cast<int64_t>(b) * num_real_nodes * 2 * D;
    const int64_t* sample_sorted =
        all_sorted + static_cast<int64_t>(b) * num_leaves;
    const int64_t index_offset = route ? 0 : static_cast<int64_t>(true_num_leaves);

    float q[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) q[d] = query_points[flat_query_idx * D + d];

    int64_t best_indices[K];
    float best_distances[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        best_indices[i] = -1;
        best_distances[i] = kFloatInf;
    }

    constexpr int stack_capacity = 32;
    unsigned long long stack[stack_capacity];
    int stack_size = 0;
    namespace tree = implicit_bvh::tree;
    const int leaf_level = tree::leaf_level(num_leaves);
    const int first_leaf = tree::first_index_at_level(leaf_level);
    int implicit_idx = 0;
    float node_dist = min_distance_sq_to_aabb<D>(q, sample_aabbs);

    while (true) {
        if (node_dist <= best_distances[K - 1]) {
            if (implicit_idx >= first_leaf) {
                insert_candidate<K>(
                    sample_sorted[implicit_idx - first_leaf] + index_offset,
                    node_dist, best_indices, best_distances);
            } else {
                const int left_idx = tree::left_child(implicit_idx);
                const int right_idx = tree::right_child(implicit_idx);
                const bool left_real = !tree::is_virtual(num_leaves, left_idx);
                const bool right_real = !tree::is_virtual(num_leaves, right_idx);
                const int left_mem = left_real ? tree::memory_index(num_leaves, left_idx) : -1;
                const int right_mem = right_real ? tree::memory_index(num_leaves, right_idx) : -1;
                const float left_dist = left_real
                    ? min_distance_sq_to_aabb<D>(q, sample_aabbs + static_cast<int64_t>(left_mem) * 2 * D)
                    : kFloatInf;
                const float right_dist = right_real
                    ? min_distance_sq_to_aabb<D>(q, sample_aabbs + static_cast<int64_t>(right_mem) * 2 * D)
                    : kFloatInf;
                const bool left_near = left_dist <= right_dist;
                const int near_idx = left_near ? left_idx : right_idx;
                const int far_idx = left_near ? right_idx : left_idx;
                const float near_dist = left_near ? left_dist : right_dist;
                const float far_dist = left_near ? right_dist : left_dist;
                const bool near_real = left_near ? left_real : right_real;
                const bool far_real = left_near ? right_real : left_real;
                const float cutoff = best_distances[K - 1];
                if (far_real && far_dist <= cutoff) {
                    stack[stack_size++] = pack_cached_stack_entry(far_idx, far_dist);
                }
                if (near_real && near_dist <= cutoff) {
                    implicit_idx = near_idx;
                    node_dist = near_dist;
                    continue;
                }
            }
        }
        if (stack_size == 0) break;
        const unsigned long long entry = stack[--stack_size];
        implicit_idx = static_cast<int>(entry >> 32);
        node_dist = __uint_as_float(static_cast<unsigned int>(entry));
    }

    const int64_t out_offset = static_cast<int64_t>(launch_query_idx) * K;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_indices[out_offset + i] = best_indices[i];
        out_distances[out_offset + i] = best_distances[i];
    }
}

template <int D, int K>
std::tuple<torch::Tensor, torch::Tensor>
launch_query_knn_routed_batched_cached_bounds_spatial(
    torch::Tensor true_node_aabbs,
    torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs,
    torch::Tensor false_sorted_indices,
    torch::Tensor query_points,
    torch::Tensor routes,
    torch::Tensor query_order,
    int true_num_leaves,
    int true_num_real_nodes,
    int false_num_leaves,
    int false_num_real_nodes
) {
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    auto indices = torch::empty({B, M, K}, true_sorted_indices.options());
    auto distances = torch::empty({B, M, K}, query_points.options());
    constexpr int threads = 256;
    const int blocks = static_cast<int>((B * M + threads - 1) / threads);
    query_knn_routed_batched_cached_bounds_spatial_kernel<D, K>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            true_node_aabbs.data_ptr<float>(), true_sorted_indices.data_ptr<int64_t>(),
            false_node_aabbs.data_ptr<float>(), false_sorted_indices.data_ptr<int64_t>(),
            query_points.data_ptr<float>(), routes.data_ptr<bool>(),
            query_order.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
            distances.data_ptr<float>(), static_cast<int>(B), static_cast<int>(M),
            true_num_leaves, true_num_real_nodes,
            false_num_leaves, false_num_real_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, distances);
}

template <int K, bool Ordered, bool SpatialOutput = false>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_dispatch_dim(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int dim
) {
    if (dim == 2) {
        return launch_query_knn_batched_cached_bounds<2, K, Ordered, SpatialOutput>(
            node_aabbs, sorted_indices, query_points, query_order,
            num_leaves, num_real_nodes);
    }
    return launch_query_knn_batched_cached_bounds<3, K, Ordered, SpatialOutput>(
        node_aabbs, sorted_indices, query_points, query_order,
        num_leaves, num_real_nodes);
}

template <bool Ordered, bool SpatialOutput = false>
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_impl(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int dim,
    int k
) {
    const char* op = Ordered
        ? "query_knn_batched_cached_bounds_ordered"
        : "query_knn_batched_cached_bounds";
    TORCH_CHECK(dim == 2 || dim == 3, op, ": D must be 2 or 3");
    TORCH_CHECK(query_points.is_cuda() && query_points.is_contiguous() &&
                query_points.scalar_type() == torch::kFloat32,
                op, ": query_points must be contiguous CUDA float32");
    TORCH_CHECK(query_points.dim() == 3 && query_points.size(2) == dim,
                op, ": query_points must have shape (B, M, D)");
    TORCH_CHECK(node_aabbs.is_cuda() && node_aabbs.is_contiguous() &&
                node_aabbs.scalar_type() == torch::kFloat32,
                op, ": node_aabbs must be contiguous CUDA float32");
    TORCH_CHECK(node_aabbs.dim() == 3 &&
                node_aabbs.size(0) == query_points.size(0) &&
                node_aabbs.size(1) == num_real_nodes && node_aabbs.size(2) == 2 * dim,
                op, ": node_aabbs has incompatible shape");
    TORCH_CHECK(sorted_indices.is_cuda() && sorted_indices.is_contiguous() &&
                sorted_indices.scalar_type() == torch::kInt64,
                op, ": sorted_indices must be contiguous CUDA int64");
    TORCH_CHECK(sorted_indices.dim() == 2 &&
                sorted_indices.size(0) == query_points.size(0) &&
                sorted_indices.size(1) == num_leaves,
                op, ": sorted_indices has incompatible shape");
    if constexpr (Ordered) {
        TORCH_CHECK(query_order.is_cuda() && query_order.is_contiguous() &&
                    query_order.scalar_type() == torch::kInt64,
                    op, ": query_order must be contiguous CUDA int64");
        TORCH_CHECK(query_order.dim() == 2 &&
                    query_order.size(0) == query_points.size(0) &&
                    query_order.size(1) == query_points.size(1),
                    op, ": query_order must have shape (B, M)");
    }
    TORCH_CHECK(num_leaves >= k, op, ": num_leaves must be >= k");
    c10::cuda::CUDAGuard device_guard(query_points.device());
    for (const auto& tensor : {node_aabbs, sorted_indices}) {
        TORCH_CHECK(tensor.device() == query_points.device(), op, ": all inputs must share a device");
    }
    if constexpr (Ordered) {
        TORCH_CHECK(query_order.device() == query_points.device(), op, ": all inputs must share a device");
    }

    switch (k) {
        case 4:
            return query_knn_batched_cached_bounds_dispatch_dim<4, Ordered, SpatialOutput>(
                node_aabbs, sorted_indices, query_points, query_order,
                num_leaves, num_real_nodes, dim);
        case 8:
            return query_knn_batched_cached_bounds_dispatch_dim<8, Ordered, SpatialOutput>(
                node_aabbs, sorted_indices, query_points, query_order,
                num_leaves, num_real_nodes, dim);
        case 16:
            return query_knn_batched_cached_bounds_dispatch_dim<16, Ordered, SpatialOutput>(
                node_aabbs, sorted_indices, query_points, query_order,
                num_leaves, num_real_nodes, dim);
        default:
            TORCH_CHECK(false, op, ": k must be 4, 8, or 16, got ", k);
    }
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int dim,
    int k
) {
    return query_knn_batched_cached_bounds_impl<false>(
        node_aabbs, sorted_indices, query_points, torch::Tensor(),
        num_leaves, num_real_nodes, dim, k);
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_ordered_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int dim,
    int k
) {
    return query_knn_batched_cached_bounds_impl<true>(
        node_aabbs, sorted_indices, query_points, query_order,
        num_leaves, num_real_nodes, dim, k);
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_spatial_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int dim,
    int k
) {
    return query_knn_batched_cached_bounds_impl<true, true>(
        node_aabbs, sorted_indices, query_points, query_order,
        num_leaves, num_real_nodes, dim, k);
}

std::tuple<torch::Tensor, torch::Tensor>
query_knn_routed_batched_cached_bounds_spatial_cuda(
    torch::Tensor true_node_aabbs,
    torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs,
    torch::Tensor false_sorted_indices,
    torch::Tensor query_points,
    torch::Tensor routes,
    torch::Tensor query_order,
    int true_num_leaves,
    int true_num_real_nodes,
    int false_num_leaves,
    int false_num_real_nodes,
    int dim,
    int k
) {
    const char* op = "query_knn_routed_batched_cached_bounds_spatial";
    TORCH_CHECK(dim == 2 || dim == 3, op, ": D must be 2 or 3");
    for (const auto& tensor : {true_node_aabbs, false_node_aabbs, query_points}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.scalar_type() == torch::kFloat32,
                    op, ": float inputs must be contiguous CUDA float32 tensors");
    }
    for (const auto& tensor : {true_sorted_indices, false_sorted_indices, query_order}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.scalar_type() == torch::kInt64,
                    op, ": index inputs must be contiguous CUDA int64 tensors");
    }
    TORCH_CHECK(routes.is_cuda() && routes.is_contiguous() &&
                routes.scalar_type() == torch::kBool,
                op, ": routes must be contiguous CUDA bool");
    TORCH_CHECK(query_points.dim() == 3 && query_points.size(2) == dim,
                op, ": query_points must have shape (B, M, D)");
    const int64_t B = query_points.size(0);
    const int64_t M = query_points.size(1);
    TORCH_CHECK(routes.dim() == 2 && routes.size(0) == B && routes.size(1) == M,
                op, ": routes must have shape (B, M)");
    TORCH_CHECK(query_order.dim() == 2 && query_order.size(0) == B && query_order.size(1) == M,
                op, ": query_order must have shape (B, M)");
    TORCH_CHECK(true_node_aabbs.dim() == 3 && true_node_aabbs.size(0) == B &&
                true_node_aabbs.size(1) == true_num_real_nodes &&
                true_node_aabbs.size(2) == 2 * dim,
                op, ": true_node_aabbs has incompatible shape");
    TORCH_CHECK(false_node_aabbs.dim() == 3 && false_node_aabbs.size(0) == B &&
                false_node_aabbs.size(1) == false_num_real_nodes &&
                false_node_aabbs.size(2) == 2 * dim,
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
    for (const auto& tensor : {true_node_aabbs, true_sorted_indices,
                               false_node_aabbs, false_sorted_indices,
                               routes, query_order}) {
        TORCH_CHECK(tensor.device() == query_points.device(),
                    op, ": all inputs must share a device");
    }

#define DISPATCH_ROUTED_CACHED(D, K) \
    return launch_query_knn_routed_batched_cached_bounds_spatial<D, K>( \
        true_node_aabbs, true_sorted_indices, false_node_aabbs, false_sorted_indices, \
        query_points, routes, query_order, true_num_leaves, true_num_real_nodes, \
        false_num_leaves, false_num_real_nodes)
    if (dim == 2 && k == 4) DISPATCH_ROUTED_CACHED(2, 4);
    if (dim == 2 && k == 8) DISPATCH_ROUTED_CACHED(2, 8);
    if (dim == 2 && k == 16) DISPATCH_ROUTED_CACHED(2, 16);
    if (dim == 3 && k == 4) DISPATCH_ROUTED_CACHED(3, 4);
    if (dim == 3 && k == 8) DISPATCH_ROUTED_CACHED(3, 8);
    TORCH_CHECK(dim == 3 && k == 16, op, ": K must be 4, 8, or 16");
    DISPATCH_ROUTED_CACHED(3, 16);
#undef DISPATCH_ROUTED_CACHED
}
