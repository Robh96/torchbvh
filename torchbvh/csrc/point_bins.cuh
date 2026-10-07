// Experimental exact 2D bin search. Included after the reference traversal.
// Every tie and every uncertified result is recomputed by the original BVH.
#pragma once

__device__ inline int point_bin_coordinate(float x, float lo, float hi, int resolution) {
    const float fraction = (x - lo) / fmaxf(hi - lo, 1.0e-20f);
    return min(resolution - 1, max(0, __float2int_rd(fraction * resolution)));
}

__global__ void build_point_bins_kernel(
    const float* points, const float* scene_min, const float* scene_max,
    int* heads, int* next, int* counts, int* unsafe, int batch, int count, int resolution) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch * count) return;
    const int b = row / count, i = row - b * count;
    const int x = point_bin_coordinate(points[row * 2], scene_min[b * 2], scene_max[b * 2], resolution);
    const int y = point_bin_coordinate(points[row * 2 + 1], scene_min[b * 2 + 1], scene_max[b * 2 + 1], resolution);
    const int cell = (b * resolution + y) * resolution + x;
    next[row] = atomicExch(heads + cell, i);
    if (atomicAdd(counts + cell, 1) >= 64) atomicExch(unsafe + b, 1);
    #pragma unroll
    for (int d = 0; d < 2; ++d) {
        const float margin = 32 * 1.1920928955078125e-7f * (fabsf(scene_min[b*2+d]) + fabsf(scene_max[b*2+d]) + 1);
        if ((scene_max[b*2+d] - scene_min[b*2+d]) / resolution <= 4 * margin)
            atomicExch(unsafe + b, 1);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_point_bins_cuda(
    torch::Tensor points, torch::Tensor scene_min, torch::Tensor scene_max, int resolution) {
    TORCH_CHECK(points.is_cuda() && points.is_contiguous() && points.scalar_type() == torch::kFloat32 &&
                points.dim() == 3 && points.size(2) == 2 && points.size(0) > 0 && points.size(1) > 0,
                "point_bins: expected contiguous CUDA float32 (B,N,2)");
    TORCH_CHECK(resolution >= 4 && resolution <= 256 && points.size(0) * points.size(1) <= INT32_MAX,
                "point_bins: invalid extent");
    for (const auto& bounds : {scene_min, scene_max})
        TORCH_CHECK(bounds.is_cuda() && bounds.is_contiguous() && bounds.device() == points.device() &&
                    bounds.scalar_type() == torch::kFloat32 && bounds.dim() == 2 &&
                    bounds.size(0) == points.size(0) && bounds.size(1) == 2, "point_bins: invalid bounds");
    c10::cuda::CUDAGuard guard(points.device());
    auto opts = points.options().dtype(torch::kInt32);
    auto heads = torch::full({points.size(0), resolution, resolution}, -1, opts);
    auto next = torch::empty({points.size(0), points.size(1)}, opts);
    auto counts = torch::zeros_like(heads);
    auto unsafe = torch::zeros({points.size(0)}, opts);
    const int total = points.size(0) * points.size(1);
    build_point_bins_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(), scene_min.data_ptr<float>(), scene_max.data_ptr<float>(),
        heads.data_ptr<int>(), next.data_ptr<int>(), counts.data_ptr<int>(), unsafe.data_ptr<int>(), points.size(0), points.size(1), resolution);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {heads, next, unsafe};
}

template <int K>
__device__ inline void search_point_cell(
    int x, int y, int resolution, const int* heads, const int* next,
    const float* points, const float* q, int64_t* best, float* distances, bool& tie, int& work) {
    if (x < 0 || x >= resolution || y < 0 || y >= resolution) return;
    int index = heads[y * resolution + x];
    while (index >= 0) {
        if (++work > 64) { tie = true; return; }
        float distance = 0;
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            const float delta = fmaxf(points[index * 2 + d] - q[d], fmaxf(0.0f, q[d] - points[index * 2 + d]));
            distance += delta * delta;
        }
        // Conservatively fall back even when a tie later leaves the top K.
        #pragma unroll
        for (int j = 0; j < K; ++j) tie |= distance == distances[j];
        insert_candidate<K>(index, distance, best, distances);
        index = next[index];
    }
}

template <int K>
__global__ void query_point_bins_kernel(
    const float* points, const float* scene_min, const float* scene_max,
    const int* heads, const int* next, const int* unsafe, const float* queries, const bool* routes,
    const int64_t* order, int64_t* indices, float* out_distances, bool* fallback,
    int B, int M, int count, int true_count, int resolution) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * M) return;
    fallback[row] = true;
    const int b = row / M, m = order[row];
    if (routes[b * M + m] || unsafe[b]) return;
    const float* q = queries + (static_cast<int64_t>(b) * M + m) * 2;
    const float* lo = scene_min + b * 2;
    const float* hi = scene_max + b * 2;
    const int cx = point_bin_coordinate(q[0], lo[0], hi[0], resolution);
    const int cy = point_bin_coordinate(q[1], lo[1], hi[1], resolution);
    const int* cell_heads = heads + b * resolution * resolution;
    const int* cell_next = next + static_cast<int64_t>(b) * count;
    const float* sample = points + static_cast<int64_t>(b) * count * 2;
    int64_t best[K]; float distances[K]; bool tie = false;
    int work = 0;
    #pragma unroll
    for (int j = 0; j < K; ++j) { best[j] = -1; distances[j] = kFloatInf; }
    bool certified = false;
    for (int r = 0; r <= 4; ++r) {
        if (r == 0) {
            search_point_cell<K>(cx, cy, resolution, cell_heads, cell_next, sample, q, best, distances, tie, work);
        } else {
            for (int x = cx - r; x <= cx + r; ++x) {
                search_point_cell<K>(x, cy - r, resolution, cell_heads, cell_next, sample, q, best, distances, tie, work);
                search_point_cell<K>(x, cy + r, resolution, cell_heads, cell_next, sample, q, best, distances, tie, work);
            }
            for (int y = cy - r + 1; y < cy + r; ++y) {
                search_point_cell<K>(cx - r, y, resolution, cell_heads, cell_next, sample, q, best, distances, tie, work);
                search_point_cell<K>(cx + r, y, resolution, cell_heads, cell_next, sample, q, best, distances, tie, work);
            }
        }
        if (tie) return;
        float outside = kFloatInf;
        // Account conservatively for float32 cell-assignment rounding. Boundaries
        // are moved toward the query; a strict cutoff also covers unseen ties.
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            const int center = d == 0 ? cx : cy;
            const float width = fmaxf(hi[d] - lo[d], 1.0e-20f) / resolution;
            const float margin = 32 * 1.1920928955078125e-7f * (fabsf(lo[d]) + fabsf(hi[d]) + 1);
            if (center - r > 0) outside = fminf(outside, fmaxf(0.0f, q[d] - (lo[d] + (center - r) * width + margin)));
            if (center + r < resolution - 1) outside = fminf(outside, fmaxf(0.0f, lo[d] + (center + r + 1) * width - margin - q[d]));
        }
        if (best[K - 1] >= 0 && distances[K - 1] < outside * outside) { certified = true; break; }
    }
    if (!certified) return;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        indices[static_cast<int64_t>(row) * K + j] = best[j] + true_count;
        out_distances[static_cast<int64_t>(row) * K + j] = distances[j];
    }
    fallback[row] = false;
}

template <int D, int K>
__global__ void query_knn_routed_bins_fallback_kernel(
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
    int false_num_real_nodes, const bool* fallback
) {
    const int launch_query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_queries = B * M;
    if (launch_query_idx >= total_queries || !fallback[launch_query_idx]) return;

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
query_knn_routed_bins_cuda(
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
    int k, torch::Tensor points, torch::Tensor scene_min, torch::Tensor scene_max,
    torch::Tensor heads, torch::Tensor next, torch::Tensor unsafe, int resolution
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

    TORCH_CHECK(dim == 2 && k == 4, "point_bins: this experimental dispatch requires D=2,K=4");
    TORCH_CHECK(resolution >= 4 && resolution <= 256, "point_bins: invalid resolution");
    for (const auto& t : {points, scene_min, scene_max})
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == query_points.device() && t.scalar_type() == torch::kFloat32,
                    "point_bins: invalid float metadata");
    for (const auto& t : {heads, next, unsafe})
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == query_points.device() && t.scalar_type() == torch::kInt32,
                    "point_bins: invalid linked-list metadata");
    TORCH_CHECK(points.dim() == 3 && points.size(0) == B && points.size(1) == false_num_leaves && points.size(2) == 2 &&
                scene_min.dim() == 2 && scene_min.size(0) == B && scene_min.size(1) == 2 && scene_max.sizes() == scene_min.sizes() &&
                heads.dim() == 3 && heads.size(0) == B && heads.size(1) == resolution && heads.size(2) == resolution &&
                next.dim() == 2 && next.size(0) == B && next.size(1) == false_num_leaves &&
                unsafe.dim() == 1 && unsafe.size(0) == B && B*M <= INT32_MAX,
                "point_bins: incompatible linked-list metadata");
    auto indices = torch::empty({B,M,k},query_order.options());
    auto distances = torch::empty({B,M,k},query_points.options());
    auto fallback = torch::empty({B,M},routes.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (B*M+255)/256;
    query_point_bins_kernel<4><<<blocks,256,0,stream>>>(points.data_ptr<float>(),scene_min.data_ptr<float>(),scene_max.data_ptr<float>(),
        heads.data_ptr<int>(),next.data_ptr<int>(),unsafe.data_ptr<int>(),query_points.data_ptr<float>(),routes.data_ptr<bool>(),query_order.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),distances.data_ptr<float>(),fallback.data_ptr<bool>(),B,M,false_num_leaves,true_num_leaves,resolution);
    query_knn_routed_bins_fallback_kernel<2,4><<<blocks,256,0,stream>>>(true_node_aabbs.data_ptr<float>(),true_sorted_indices.data_ptr<int64_t>(),
        false_node_aabbs.data_ptr<float>(),false_sorted_indices.data_ptr<int64_t>(),query_points.data_ptr<float>(),routes.data_ptr<bool>(),
        query_order.data_ptr<int64_t>(),indices.data_ptr<int64_t>(),distances.data_ptr<float>(),B,M,true_num_leaves,true_num_real_nodes,
        false_num_leaves,false_num_real_nodes,fallback.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {indices,distances,fallback};
}
