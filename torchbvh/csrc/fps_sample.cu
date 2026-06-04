#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "geometry.cuh"
#include "implicit_tree.cuh"

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

template <int D>
__device__ inline float point_distance_sq(
    const float* __restrict__ points,
    int64_t base,
    int i,
    int j
) {
    float dist = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float delta = points[base + static_cast<int64_t>(i) * D + d]
            - points[base + static_cast<int64_t>(j) * D + d];
        dist += delta * delta;
    }
    return dist;
}
// ---------------------------------------------------------------------------
// Exact full-scan fallback
// ---------------------------------------------------------------------------




template <int D, int THREADS>
__global__ void fps_exact_full_scan_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ seed_indices,
    int64_t* __restrict__ fps_idx,
    int* __restrict__ nearest_anchor,
    float* __restrict__ nearest_dist_sq,
    int N,
    int M
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t state_base = static_cast<int64_t>(b) * N;
    const int seed = static_cast<int>(seed_indices[b]);
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float warp_dist[THREADS / 32];
    __shared__ int warp_idx[THREADS / 32];
    __shared__ int anchor_idx;

    if (tid == 0) {
        fps_idx[static_cast<int64_t>(b) * M] = seed;
    }
    for (int i = tid; i < N; i += THREADS) {
        nearest_dist_sq[state_base + i] = point_distance_sq<D>(points, point_base, i, seed);
        nearest_anchor[state_base + i] = 0;
    }
    __syncthreads();

    for (int round_idx = 1; round_idx < M; ++round_idx) {
        float best_dist = -1.0f;
        int best_idx = 0;
        for (int i = tid; i < N; i += THREADS) {
            const float value = nearest_dist_sq[state_base + i];
            if (value > best_dist || (value == best_dist && i < best_idx)) {
                best_dist = value;
                best_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
            const int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_dist > best_dist || (other_dist == best_dist && other_idx < best_idx)) {
                best_dist = other_dist;
                best_idx = other_idx;
            }
        }
        if (lane == 0) {
            warp_dist[warp_id] = best_dist;
            warp_idx[warp_id] = best_idx;
        }
        __syncthreads();

        if (warp_id == 0) {
            best_dist = (lane < THREADS / 32) ? warp_dist[lane] : -1.0f;
            best_idx = (lane < THREADS / 32) ? warp_idx[lane] : 0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
                const int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
                if (other_dist > best_dist || (other_dist == best_dist && other_idx < best_idx)) {
                    best_dist = other_dist;
                    best_idx = other_idx;
                }
            }
            if (lane == 0) {
                anchor_idx = best_idx;
                fps_idx[static_cast<int64_t>(b) * M + round_idx] = static_cast<int64_t>(best_idx);
            }
        }
        __syncthreads();

        for (int i = tid; i < N; i += THREADS) {
            const float d_new = point_distance_sq<D>(points, point_base, i, anchor_idx);
            float& current = nearest_dist_sq[state_base + i];
            if (d_new < current) {
                current = d_new;
                nearest_anchor[state_base + i] = round_idx;
            }
        }
        __syncthreads();
    }
}





template <int D>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_exact_full_scan_impl(
    torch::Tensor points,
    torch::Tensor seed_indices,
    int M
) {
    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    auto fps_idx = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt64));
    auto nearest_anchor = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
    auto nearest_dist_sq = torch::empty({B, N}, points.options());

    constexpr int threads = 256;
    fps_exact_full_scan_kernel<D, threads><<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        seed_indices.data_ptr<int64_t>(),
        fps_idx.data_ptr<int64_t>(),
        nearest_anchor.data_ptr<int>(),
        nearest_dist_sq.data_ptr<float>(),
        N,
        M
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(fps_idx, nearest_anchor, nearest_dist_sq);
}

// ---------------------------------------------------------------------------
// Metadata helpers shared by maintained FPS routes
// ---------------------------------------------------------------------------

__global__ void fps_metadata_scatter_kernel(
    const int* __restrict__ nearest_anchor,
    const float* __restrict__ nearest_dist_sq,
    int* __restrict__ anchor_counts,
    float* __restrict__ anchor_radius,
    int B,
    int N,
    int M
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N;
    if (global_idx >= total) {
        return;
    }

    const int b = global_idx / N;
    const int anchor = nearest_anchor[global_idx];
    if (anchor < 0 || anchor >= M) {
        return;
    }

    const int out_idx = b * M + anchor;
    atomicAdd(anchor_counts + out_idx, 1);
    // Squared distances are nonnegative float32 values, so IEEE bit ordering
    // matches numeric ordering and int atomicMax is safe for this reduction.
    atomicMax(reinterpret_cast<int*>(anchor_radius + out_idx), __float_as_int(nearest_dist_sq[global_idx]));
}

__global__ void fps_reverse_leaf_pos_kernel(
    const int64_t* __restrict__ sorted_indices,
    int* __restrict__ reverse_leaf_pos,
    int B,
    int N
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * N;
    if (global_idx >= total) {
        return;
    }

    const int b = global_idx / N;
    const int leaf_pos = global_idx - b * N;
    const int original_idx = static_cast<int>(sorted_indices[global_idx]);
    reverse_leaf_pos[static_cast<int64_t>(b) * N + original_idx] = leaf_pos;
}

__global__ void fps_leaf_pos_gather_kernel(
    const int64_t* __restrict__ fps_idx,
    const int* __restrict__ reverse_leaf_pos,
    int64_t* __restrict__ fps_leaf_pos,
    int B,
    int N,
    int M
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * M;
    if (global_idx >= total) {
        return;
    }

    const int b = global_idx / M;
    const int m = global_idx - b * M;
    const int original_idx = static_cast<int>(fps_idx[global_idx]);
    fps_leaf_pos[global_idx] = static_cast<int64_t>(
        reverse_leaf_pos[static_cast<int64_t>(b) * N + original_idx]
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_metadata_cuda(
    torch::Tensor points,
    torch::Tensor fps_idx,
    torch::Tensor nearest_anchor,
    torch::Tensor nearest_dist_sq,
    torch::Tensor sorted_indices
) {
    TORCH_CHECK(points.is_cuda(), "fps_metadata: points must be a CUDA tensor");
    TORCH_CHECK(fps_idx.is_cuda(), "fps_metadata: fps_idx must be a CUDA tensor");
    TORCH_CHECK(nearest_anchor.is_cuda(), "fps_metadata: nearest_anchor must be a CUDA tensor");
    TORCH_CHECK(nearest_dist_sq.is_cuda(), "fps_metadata: nearest_dist_sq must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "fps_metadata: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "fps_metadata: points must be contiguous");
    TORCH_CHECK(fps_idx.is_contiguous(), "fps_metadata: fps_idx must be contiguous");
    TORCH_CHECK(nearest_anchor.is_contiguous(), "fps_metadata: nearest_anchor must be contiguous");
    TORCH_CHECK(nearest_dist_sq.is_contiguous(), "fps_metadata: nearest_dist_sq must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "fps_metadata: sorted_indices must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "fps_metadata: points must be float32");
    TORCH_CHECK(fps_idx.scalar_type() == torch::kInt64, "fps_metadata: fps_idx must be int64");
    TORCH_CHECK(nearest_anchor.scalar_type() == torch::kInt32, "fps_metadata: nearest_anchor must be int32");
    TORCH_CHECK(nearest_dist_sq.scalar_type() == torch::kFloat32, "fps_metadata: nearest_dist_sq must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "fps_metadata: sorted_indices must be int64");
    TORCH_CHECK(points.dim() == 3, "fps_metadata: points must have shape (B, N, D)");
    TORCH_CHECK(fps_idx.dim() == 2, "fps_metadata: fps_idx must have shape (B, M)");
    TORCH_CHECK(nearest_anchor.dim() == 2, "fps_metadata: nearest_anchor must have shape (B, N)");
    TORCH_CHECK(nearest_dist_sq.dim() == 2, "fps_metadata: nearest_dist_sq must have shape (B, N)");
    TORCH_CHECK(sorted_indices.dim() == 2, "fps_metadata: sorted_indices must have shape (B, N)");

    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int M = static_cast<int>(fps_idx.size(1));
    TORCH_CHECK(points.size(2) == 2 || points.size(2) == 3, "fps_metadata: D must be 2 or 3");
    TORCH_CHECK(fps_idx.size(0) == B, "fps_metadata: fps_idx batch size must match points");
    TORCH_CHECK(nearest_anchor.size(0) == B && nearest_anchor.size(1) == N, "fps_metadata: nearest_anchor shape must match points");
    TORCH_CHECK(nearest_dist_sq.size(0) == B && nearest_dist_sq.size(1) == N, "fps_metadata: nearest_dist_sq shape must match points");
    TORCH_CHECK(sorted_indices.size(0) == B && sorted_indices.size(1) == N, "fps_metadata: sorted_indices shape must match points");
    TORCH_CHECK(M >= 1 && M <= N, "fps_metadata: M must be in [1, N]");

    c10::cuda::CUDAGuard device_guard(points.device());
    TORCH_CHECK(fps_idx.device() == points.device(), "fps_metadata: fps_idx and points must be on the same device");
    TORCH_CHECK(nearest_anchor.device() == points.device(), "fps_metadata: nearest_anchor and points must be on the same device");
    TORCH_CHECK(nearest_dist_sq.device() == points.device(), "fps_metadata: nearest_dist_sq and points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == points.device(), "fps_metadata: sorted_indices and points must be on the same device");

    auto anchor_counts = torch::zeros({B, M}, nearest_anchor.options().dtype(torch::kInt32));
    auto anchor_radius = torch::zeros({B, M}, points.options());
    auto reverse_leaf_pos = torch::empty({B, N}, nearest_anchor.options().dtype(torch::kInt32));
    auto fps_leaf_pos = torch::empty({B, M}, fps_idx.options().dtype(torch::kInt64));

    constexpr int threads = 256;
    const int point_blocks = (B * N + threads - 1) / threads;
    fps_metadata_scatter_kernel<<<point_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        nearest_anchor.data_ptr<int>(),
        nearest_dist_sq.data_ptr<float>(),
        anchor_counts.data_ptr<int>(),
        anchor_radius.data_ptr<float>(),
        B,
        N,
        M
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fps_reverse_leaf_pos_kernel<<<point_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        sorted_indices.data_ptr<int64_t>(),
        reverse_leaf_pos.data_ptr<int>(),
        B,
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int anchor_blocks = (B * M + threads - 1) / threads;
    fps_leaf_pos_gather_kernel<<<anchor_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        fps_idx.data_ptr<int64_t>(),
        reverse_leaf_pos.data_ptr<int>(),
        fps_leaf_pos.data_ptr<int64_t>(),
        B,
        N,
        M
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(anchor_radius, anchor_counts, fps_leaf_pos);
}

__global__ void fps_scatter_leaf_state_kernel(
    const int64_t* __restrict__ sorted_indices,
    const int* __restrict__ leaf_nearest_anchor,
    const float* __restrict__ leaf_nearest_dist_sq,
    int* __restrict__ nearest_anchor,
    float* __restrict__ nearest_dist_sq,
    int B,
    int N
) {
    const int64_t global_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(B) * N;
    if (global_idx >= total) {
        return;
    }

    const int b = static_cast<int>(global_idx / N);
    const int original_idx = static_cast<int>(sorted_indices[global_idx]);
    const int64_t out_idx = static_cast<int64_t>(b) * N + original_idx;
    nearest_anchor[out_idx] = leaf_nearest_anchor[global_idx];
    nearest_dist_sq[out_idx] = leaf_nearest_dist_sq[global_idx];
}

// ---------------------------------------------------------------------------
// Approximate bucket-queue graph route and workspace cache
// ---------------------------------------------------------------------------

struct BucketQueueWorkspaceKey {
    int B, N, D, M;
    int bucket_count;
    int effective_refresh_interval;
    int dirty_refresh_interval;
    int max_iterations;
    int max_commit_per_iteration;
    int candidates_per_round;
    int anchors_per_round;
    int top_risk_buckets;
    int threads_per_block;

    bool operator==(const BucketQueueWorkspaceKey& o) const noexcept {
        return B==o.B && N==o.N && D==o.D && M==o.M
            && bucket_count==o.bucket_count
            && effective_refresh_interval==o.effective_refresh_interval
            && dirty_refresh_interval==o.dirty_refresh_interval
            && max_iterations==o.max_iterations
            && max_commit_per_iteration==o.max_commit_per_iteration
            && candidates_per_round==o.candidates_per_round
            && anchors_per_round==o.anchors_per_round
            && top_risk_buckets==o.top_risk_buckets
            && threads_per_block==o.threads_per_block;
    }
};

struct BucketQueueWorkspaceKeyHash {
    std::size_t operator()(const BucketQueueWorkspaceKey& k) const noexcept {
        std::size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(k.B); mix(k.N); mix(k.D); mix(k.M);
        mix(k.bucket_count); mix(k.effective_refresh_interval);
        mix(k.dirty_refresh_interval); mix(k.max_iterations);
        mix(k.max_commit_per_iteration); mix(k.candidates_per_round);
        mix(k.anchors_per_round); mix(k.top_risk_buckets);
        mix(k.threads_per_block);
        return h;
    }
};

struct BucketQueueWorkspace {
    torch::Tensor ws_points, ws_sorted_indices, ws_seed_indices;
    torch::Tensor fps_idx, nearest_anchor, nearest_dist_sq;
    torch::Tensor leaf_nearest_anchor, leaf_nearest_dist_sq;
    torch::Tensor selected_flags, selected_counts;
    torch::Tensor bucket_leaf_begin, bucket_leaf_end, bucket_last_applied;
    torch::Tensor bucket_max_dist_sq, bucket_max_leaf, bucket_aabb, dirty_flags;
    torch::Tensor mode_flags, visited_leaf_counts, committed_counts;
    torch::Tensor candidate_counts, rejected_counts, refresh_flags;

    cudaStream_t ws_stream = nullptr;
    cudaEvent_t ws_input_ready = nullptr;
    cudaEvent_t ws_output_ready = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    ~BucketQueueWorkspace() {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        if (ws_output_ready) { cudaEventDestroy(ws_output_ready); ws_output_ready = nullptr; }
        if (ws_input_ready) { cudaEventDestroy(ws_input_ready); ws_input_ready = nullptr; }
        if (ws_stream) { cudaStreamDestroy(ws_stream); ws_stream = nullptr; }
    }
};

static std::unordered_map<
    BucketQueueWorkspaceKey,
    std::shared_ptr<BucketQueueWorkspace>,
    BucketQueueWorkspaceKeyHash
> s_bucket_queue_workspace_cache;
static std::mutex s_bucket_queue_workspace_mutex;

template <int D, int THREADS>
__global__ void fps_bucket_queue_init_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ seed_indices,
    const int64_t* __restrict__ sorted_indices,
    int64_t* __restrict__ fps_idx,
    int* __restrict__ selected_flags,
    int* __restrict__ selected_counts,
    float* __restrict__ leaf_nearest_dist_sq,
    int* __restrict__ leaf_nearest_anchor,
    int* __restrict__ bucket_leaf_begin,
    int* __restrict__ bucket_leaf_end,
    int* __restrict__ bucket_last_applied,
    float* __restrict__ bucket_max_dist_sq,
    int* __restrict__ bucket_max_leaf,
    float* __restrict__ bucket_aabb,
    int B,
    int N,
    int M,
    int leaf_level,
    int bucket_level,
    int bucket_count
) {
    namespace tree = implicit_bvh::tree;
    const int b = blockIdx.y;
    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || bucket >= bucket_count) {
        return;
    }

    const int first_bucket = tree::first_index_at_level(bucket_level);
    const int bucket_implicit = first_bucket + bucket;
    const int descend = leaf_level - bucket_level;
    const int first_leaf_implicit = tree::descendant(bucket_implicit, descend, 0);
    const int last_leaf_implicit = tree::descendant(bucket_implicit, descend, (1 << descend) - 1);
    const int first_leaf = tree::first_index_at_level(leaf_level);
    const int leaf_begin = max(0, first_leaf_implicit - first_leaf);
    const int leaf_end = min(N, last_leaf_implicit - first_leaf + 1);
    const int64_t bucket_idx = static_cast<int64_t>(b) * bucket_count + bucket;

    if (tid == 0) {
        bucket_leaf_begin[bucket_idx] = leaf_begin;
        bucket_leaf_end[bucket_idx] = leaf_end;
        bucket_last_applied[bucket_idx] = 1;
        if (bucket == 0) {
            const int seed = static_cast<int>(seed_indices[b]);
            fps_idx[static_cast<int64_t>(b) * M] = seed;
            selected_counts[b] = 1;
            selected_flags[static_cast<int64_t>(b) * N + seed] = 1;
        }
    }

    const int seed = static_cast<int>(seed_indices[b]);
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;
    float local_best = -1.0f;
    int local_leaf = leaf_begin;
    int local_original = N;
    float local_min[3] = {1e30f, 1e30f, 1e30f};
    float local_max[3] = {-1e30f, -1e30f, -1e30f};
    for (int leaf = leaf_begin + tid; leaf < leaf_end; leaf += THREADS) {
        const int original = static_cast<int>(sorted_indices[leaf_base + leaf]);
        const float dist = point_distance_sq<D>(points, point_base, original, seed);
        leaf_nearest_dist_sq[leaf_base + leaf] = dist;
        leaf_nearest_anchor[leaf_base + leaf] = 0;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const float v = points[point_base + static_cast<int64_t>(original) * D + d];
            if (v < local_min[d]) local_min[d] = v;
            if (v > local_max[d]) local_max[d] = v;
        }
        const bool better = dist > local_best
            || (dist == local_best && selected_flags[static_cast<int64_t>(b) * N + original] == 0
                && (local_original == seed || original < local_original));
        if (better) {
            local_best = dist;
            local_leaf = leaf;
            local_original = original;
        }
    }

    __shared__ float shared_dist[THREADS];
    __shared__ int shared_leaf[THREADS];
    __shared__ int shared_original[THREADS];
    __shared__ float sh_aabb_min[THREADS * 3];
    __shared__ float sh_aabb_max[THREADS * 3];
    shared_dist[tid] = local_best;
    shared_leaf[tid] = local_leaf;
    shared_original[tid] = local_original;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sh_aabb_min[tid * 3 + d] = local_min[d];
        sh_aabb_max[tid * 3 + d] = local_max[d];
    }
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_dist = shared_dist[tid + stride];
            const int other_original = shared_original[tid + stride];
            if (other_dist > shared_dist[tid]
                    || (other_dist == shared_dist[tid] && other_original < shared_original[tid])) {
                shared_dist[tid] = other_dist;
                shared_leaf[tid] = shared_leaf[tid + stride];
                shared_original[tid] = other_original;
            }
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                if (sh_aabb_min[(tid + stride) * 3 + d] < sh_aabb_min[tid * 3 + d])
                    sh_aabb_min[tid * 3 + d] = sh_aabb_min[(tid + stride) * 3 + d];
                if (sh_aabb_max[(tid + stride) * 3 + d] > sh_aabb_max[tid * 3 + d])
                    sh_aabb_max[tid * 3 + d] = sh_aabb_max[(tid + stride) * 3 + d];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        bucket_max_dist_sq[bucket_idx] = shared_dist[0];
        bucket_max_leaf[bucket_idx] = shared_leaf[0];
        const int64_t aabb_base = bucket_idx * 2 * D;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            bucket_aabb[aabb_base + d]     = sh_aabb_min[d];
            bucket_aabb[aabb_base + D + d] = sh_aabb_max[d];
        }
    }
}

// Tests newly committed anchors against each bucket's AABB and marks dirty only when
// min_dist_sq(anchor, aabb) < bucket_max_dist_sq — so the refresh kernel skips
// geometrically-irrelevant buckets entirely.
template <int D, int THREADS>
__global__ void fps_mark_dirty_aabb_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ fps_idx,
    const int* __restrict__ selected_flags,
    const float* __restrict__ bucket_aabb,
    const float* __restrict__ bucket_max_dist_sq,
    int* __restrict__ dirty_flags,
    int B, int N, int M, int bucket_count,
    int committed_start, int max_commit_per_iter
) {
    const int bucket = blockIdx.x;
    const int b     = blockIdx.y;
    if (b >= B || bucket >= bucket_count) return;
    if (threadIdx.x != 0) return;

    const int64_t bucket_idx = static_cast<int64_t>(b) * bucket_count + bucket;
    const float max_dsq = bucket_max_dist_sq[bucket_idx];
    if (max_dsq < 0.0f) return;

    const float* aabb        = bucket_aabb + bucket_idx * 2 * D;
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t anchor_base = static_cast<int64_t>(b) * M;
    const int64_t sel_base   = static_cast<int64_t>(b) * N;
    const int committed_end  = min(committed_start + max_commit_per_iter, M);

    for (int slot = committed_start; slot < committed_end; ++slot) {
        const int orig = static_cast<int>(fps_idx[anchor_base + slot]);
        if (orig < 0 || orig >= N) continue;
        if (selected_flags[sel_base + orig] == 0) continue;
        const float* q = points + point_base + static_cast<int64_t>(orig) * D;
        const float dsq = min_distance_sq_to_aabb<D>(q, aabb);
        if (dsq < max_dsq) {
            dirty_flags[bucket_idx] = 1;
            return;
        }
    }
}

template <int THREADS, int MAX_CANDIDATES, int MAX_R>
__global__ void fps_bucket_queue_select_kernel(
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ bucket_max_dist_sq,
    const int* __restrict__ bucket_max_leaf,
    int64_t* __restrict__ fps_idx,
    int* __restrict__ selected_flags,
    int* __restrict__ selected_counts,
    int* __restrict__ mode_flags,
    int* __restrict__ visited_leaf_counts,
    int* __restrict__ committed_counts,
    int* __restrict__ candidate_counts,
    int* __restrict__ rejected_counts,
    int* __restrict__ dirty_flags,
    int B,
    int N,
    int M,
    int bucket_count,
    int candidates_per_round,
    int anchors_per_round,
    int top_risk_buckets,
    float alpha,
    int stat_slot
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    (void)dirty_flags;
    (void)top_risk_buckets;
    if (b >= B || selected_counts[b] >= M) {
        return;
    }

    __shared__ float cand_dist[MAX_CANDIDATES];
    __shared__ int cand_original[MAX_CANDIDATES];
    __shared__ float warp_best_dist[THREADS / 32];
    __shared__ int warp_best_original[THREADS / 32];
    __shared__ int warp_best_bucket[THREADS / 32];
    __shared__ int picked_bucket_sh;

    if (tid < MAX_CANDIDATES) {
        cand_dist[tid]     = -1.0f;
        cand_original[tid] = N;
    }

    const int64_t bucket_base = static_cast<int64_t>(b) * bucket_count;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;

    // Load per-thread local state: each thread covers one bucket (tid < bucket_count).
    float local_dist     = -1.0f;
    int   local_original = N;
    int   local_bucket   = -1;
    if (tid < bucket_count) {
        const float d  = bucket_max_dist_sq[bucket_base + tid];
        const int lf   = bucket_max_leaf[bucket_base + tid];
        if (lf >= 0 && lf < N) {
            const int orig = static_cast<int>(sorted_indices[leaf_base + lf]);
            if (selected_flags[static_cast<int64_t>(b) * N + orig] == 0) {
                local_dist     = d;
                local_original = orig;
                local_bucket   = tid;
            }
        }
    }
    __syncthreads();

    // Warp-parallel iterative top-K: each iteration finds the global best remaining
    // bucket using the the exact FPS top-distance comparator, then
    // invalidates that bucket so the next iteration skips it.
    for (int slot = 0; slot < candidates_per_round; ++slot) {
        float wd = local_dist;
        int   wo = local_original;
        int   wb = local_bucket;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float od = __shfl_down_sync(0xffffffff, wd, offset);
            const int   oo = __shfl_down_sync(0xffffffff, wo, offset);
            const int   ob = __shfl_down_sync(0xffffffff, wb, offset);
            if (od > wd || (od == wd && oo < wo)) {
                wd = od; wo = oo; wb = ob;
            }
        }
        if (lane == 0) {
            warp_best_dist[warp_id]     = wd;
            warp_best_original[warp_id] = wo;
            warp_best_bucket[warp_id]   = wb;
        }
        __syncthreads();

        if (warp_id == 0) {
            float fd = (lane < THREADS / 32) ? warp_best_dist[lane]     : -1.0f;
            int   fo = (lane < THREADS / 32) ? warp_best_original[lane] : N;
            int   fb = (lane < THREADS / 32) ? warp_best_bucket[lane]   : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float od = __shfl_down_sync(0xffffffff, fd, offset);
                const int   oo = __shfl_down_sync(0xffffffff, fo, offset);
                const int   ob = __shfl_down_sync(0xffffffff, fb, offset);
                if (od > fd || (od == fd && oo < fo)) {
                    fd = od; fo = oo; fb = ob;
                }
            }
            if (lane == 0) {
                cand_dist[slot]     = fd;
                cand_original[slot] = fo;
                picked_bucket_sh    = fb;
            }
        }
        __syncthreads();

        if (local_bucket == picked_bucket_sh) {
            local_dist     = -1.0f;
            local_original = N;
            local_bucket   = -1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int count = selected_counts[b];
        int commits = 0;
        int rejects = 0;
        int candidates = 0;
        const float reject_threshold = alpha * alpha * max(0.0f, cand_dist[0]);
        for (int slot = 0; slot < candidates_per_round && commits < anchors_per_round && count < M; ++slot) {
            const int original = cand_original[slot];
            if (original < 0 || original >= N) {
                continue;
            }
            candidates += 1;
            if (selected_flags[static_cast<int64_t>(b) * N + original] != 0) {
                rejects += 1;
                continue;
            }
            if (alpha > 0.0f && cand_dist[slot] <= reject_threshold && count + anchors_per_round < M) {
                rejects += 1;
                continue;
            }
            fps_idx[static_cast<int64_t>(b) * M + count] = static_cast<int64_t>(original);
            selected_flags[static_cast<int64_t>(b) * N + original] = 1;
            count += 1;
            commits += 1;
        }
        selected_counts[b] = count;
        const int64_t stat_idx = static_cast<int64_t>(b) * M + min(stat_slot, M - 1);
        mode_flags[stat_idx] = 3;
        visited_leaf_counts[stat_idx] = bucket_count;
        committed_counts[stat_idx] = commits;
        candidate_counts[stat_idx] = candidates;
        rejected_counts[stat_idx] = rejects;
    }
}

template <int D, int THREADS>
__global__ void fps_bucket_queue_refresh_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ fps_idx,
    const int* __restrict__ selected_flags,
    const int* __restrict__ selected_counts,
    float* __restrict__ leaf_nearest_dist_sq,
    int* __restrict__ leaf_nearest_anchor,
    const int* __restrict__ bucket_leaf_begin,
    const int* __restrict__ bucket_leaf_end,
    int* __restrict__ bucket_last_applied,
    float* __restrict__ bucket_max_dist_sq,
    int* __restrict__ bucket_max_leaf,
    int* __restrict__ dirty_flags,
    int* __restrict__ refresh_flags,
    int B,
    int N,
    int M,
    int bucket_count,
    bool force_refresh,
    int stat_slot
) {
    const int b = blockIdx.y;
    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || bucket >= bucket_count) {
        return;
    }

    const int64_t bucket_idx = static_cast<int64_t>(b) * bucket_count + bucket;
    if (!force_refresh && dirty_flags[bucket_idx] == 0) {
        return;
    }
    const int leaf_begin = bucket_leaf_begin[bucket_idx];
    const int leaf_end = bucket_leaf_end[bucket_idx];
    const int begin_anchor = bucket_last_applied[bucket_idx];
    const int end_anchor = selected_counts[b];
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;
    const int64_t anchor_base = static_cast<int64_t>(b) * M;

    float local_best = -1.0f;
    int local_leaf = leaf_begin;
    int local_original = N;
    for (int leaf = leaf_begin + tid; leaf < leaf_end; leaf += THREADS) {
        const int original = static_cast<int>(sorted_indices[leaf_base + leaf]);
        float current = leaf_nearest_dist_sq[leaf_base + leaf];
        int current_anchor = leaf_nearest_anchor[leaf_base + leaf];
        for (int anchor_slot = begin_anchor; anchor_slot < end_anchor; ++anchor_slot) {
            const int anchor_idx = static_cast<int>(fps_idx[anchor_base + anchor_slot]);
            const float dist = point_distance_sq<D>(points, point_base, original, anchor_idx);
            if (dist < current) {
                current = dist;
                current_anchor = anchor_slot;
            }
        }
        leaf_nearest_dist_sq[leaf_base + leaf] = current;
        leaf_nearest_anchor[leaf_base + leaf] = current_anchor;
        const bool selected = selected_flags[static_cast<int64_t>(b) * N + original] != 0;
        if (!selected && (current > local_best || (current == local_best && original < local_original))) {
            local_best = current;
            local_leaf = leaf;
            local_original = original;
        }
    }

    __shared__ float shared_dist[THREADS];
    __shared__ int shared_leaf[THREADS];
    __shared__ int shared_original[THREADS];
    shared_dist[tid] = local_best;
    shared_leaf[tid] = local_leaf;
    shared_original[tid] = local_original;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_dist = shared_dist[tid + stride];
            const int other_original = shared_original[tid + stride];
            if (other_dist > shared_dist[tid]
                    || (other_dist == shared_dist[tid] && other_original < shared_original[tid])) {
                shared_dist[tid] = other_dist;
                shared_leaf[tid] = shared_leaf[tid + stride];
                shared_original[tid] = other_original;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        bucket_last_applied[bucket_idx] = end_anchor;
        bucket_max_dist_sq[bucket_idx] = shared_dist[0];
        bucket_max_leaf[bucket_idx] = shared_leaf[0];
        dirty_flags[bucket_idx] = 0;
        if (bucket == 0) {
            refresh_flags[static_cast<int64_t>(b) * M + min(stat_slot, M - 1)] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Exact bucketed graph route kernels
// ---------------------------------------------------------------------------

template <int D, int THREADS>
__global__ void fps_exact_bucketed_init_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ seed_indices,
    const int64_t* __restrict__ sorted_indices,
    int64_t* __restrict__ fps_idx,
    float* __restrict__ leaf_nearest_dist_sq,
    int* __restrict__ leaf_nearest_anchor,
    int* __restrict__ bucket_leaf_begin,
    int* __restrict__ bucket_leaf_end,
    float* __restrict__ bucket_max_dist_sq,
    int* __restrict__ bucket_max_leaf,
    float* __restrict__ bucket_aabb,
    int* __restrict__ refreshed_counts,
    int B,
    int N,
    int M,
    int leaf_level,
    int bucket_level,
    int bucket_count,
    int count_stride
) {
    namespace tree = implicit_bvh::tree;
    const int b = blockIdx.y;
    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || bucket >= bucket_count) {
        return;
    }

    const int first_bucket = tree::first_index_at_level(bucket_level);
    const int bucket_implicit = first_bucket + bucket;
    const int descend = leaf_level - bucket_level;
    const int first_leaf_implicit = tree::descendant(bucket_implicit, descend, 0);
    const int last_leaf_implicit = tree::descendant(bucket_implicit, descend, (1 << descend) - 1);
    const int first_leaf = tree::first_index_at_level(leaf_level);
    const int leaf_begin = max(0, first_leaf_implicit - first_leaf);
    const int leaf_end = min(N, last_leaf_implicit - first_leaf + 1);
    const int64_t bucket_idx = static_cast<int64_t>(b) * bucket_count + bucket;

    if (tid == 0) {
        bucket_leaf_begin[bucket_idx] = leaf_begin;
        bucket_leaf_end[bucket_idx] = leaf_end;
        if (bucket == 0) {
            fps_idx[static_cast<int64_t>(b) * M] = seed_indices[b];
        }
        const int64_t count_idx = (count_stride == 0)
            ? static_cast<int64_t>(b)
            : static_cast<int64_t>(b) * count_stride;
        atomicAdd(refreshed_counts + count_idx, 1);
    }

    const int seed = static_cast<int>(seed_indices[b]);
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;
    float local_best = -1.0f;
    int local_leaf = leaf_begin;
    int local_original = N;
    float local_min[3] = {1e30f, 1e30f, 1e30f};
    float local_max[3] = {-1e30f, -1e30f, -1e30f};

    for (int leaf = leaf_begin + tid; leaf < leaf_end; leaf += THREADS) {
        const int original = static_cast<int>(sorted_indices[leaf_base + leaf]);
        const float dist = point_distance_sq<D>(points, point_base, original, seed);
        leaf_nearest_dist_sq[leaf_base + leaf] = dist;
        leaf_nearest_anchor[leaf_base + leaf] = 0;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const float v = points[point_base + static_cast<int64_t>(original) * D + d];
            if (v < local_min[d]) local_min[d] = v;
            if (v > local_max[d]) local_max[d] = v;
        }
        if (dist > local_best || (dist == local_best && original < local_original)) {
            local_best = dist;
            local_leaf = leaf;
            local_original = original;
        }
    }

    __shared__ float shared_dist[THREADS];
    __shared__ int shared_leaf[THREADS];
    __shared__ int shared_original[THREADS];
    __shared__ float sh_aabb_min[THREADS * 3];
    __shared__ float sh_aabb_max[THREADS * 3];
    shared_dist[tid] = local_best;
    shared_leaf[tid] = local_leaf;
    shared_original[tid] = local_original;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sh_aabb_min[tid * 3 + d] = local_min[d];
        sh_aabb_max[tid * 3 + d] = local_max[d];
    }
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_dist = shared_dist[tid + stride];
            const int other_original = shared_original[tid + stride];
            if (other_dist > shared_dist[tid]
                    || (other_dist == shared_dist[tid] && other_original < shared_original[tid])) {
                shared_dist[tid] = other_dist;
                shared_leaf[tid] = shared_leaf[tid + stride];
                shared_original[tid] = other_original;
            }
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                const float other_min = sh_aabb_min[(tid + stride) * 3 + d];
                const float other_max = sh_aabb_max[(tid + stride) * 3 + d];
                if (other_min < sh_aabb_min[tid * 3 + d]) sh_aabb_min[tid * 3 + d] = other_min;
                if (other_max > sh_aabb_max[tid * 3 + d]) sh_aabb_max[tid * 3 + d] = other_max;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        bucket_max_dist_sq[bucket_idx] = shared_dist[0];
        bucket_max_leaf[bucket_idx] = shared_leaf[0];
        const int64_t aabb_base = bucket_idx * 2 * D;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            bucket_aabb[aabb_base + d] = sh_aabb_min[d];
            bucket_aabb[aabb_base + D + d] = sh_aabb_max[d];
        }
    }
}

template <int THREADS>
__global__ void fps_exact_bucketed_select_kernel(
    const int64_t* __restrict__ sorted_indices,
    const float* __restrict__ bucket_max_dist_sq,
    const int* __restrict__ bucket_max_leaf,
    int64_t* __restrict__ fps_idx,
    int B,
    int N,
    int M,
    int bucket_count,
    int round_idx
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (b >= B || round_idx >= M) {
        return;
    }

    const int64_t bucket_base = static_cast<int64_t>(b) * bucket_count;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;
    float best_dist = -1.0f;
    int best_original = N;

    for (int bucket = tid; bucket < bucket_count; bucket += THREADS) {
        const float dist = bucket_max_dist_sq[bucket_base + bucket];
        const int leaf = bucket_max_leaf[bucket_base + bucket];
        int original = N;
        if (leaf >= 0 && leaf < N) {
            original = static_cast<int>(sorted_indices[leaf_base + leaf]);
        }
        if (dist > best_dist || (dist == best_dist && original < best_original)) {
            best_dist = dist;
            best_original = original;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
        const int other_original = __shfl_down_sync(0xffffffff, best_original, offset);
        if (other_dist > best_dist || (other_dist == best_dist && other_original < best_original)) {
            best_dist = other_dist;
            best_original = other_original;
        }
    }

    __shared__ float warp_dist[THREADS / 32];
    __shared__ int warp_original[THREADS / 32];
    if (lane == 0) {
        warp_dist[warp_id] = best_dist;
        warp_original[warp_id] = best_original;
    }
    __syncthreads();

    if (warp_id == 0) {
        best_dist = (lane < THREADS / 32) ? warp_dist[lane] : -1.0f;
        best_original = (lane < THREADS / 32) ? warp_original[lane] : N;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
            const int other_original = __shfl_down_sync(0xffffffff, best_original, offset);
            if (other_dist > best_dist || (other_dist == best_dist && other_original < best_original)) {
                best_dist = other_dist;
                best_original = other_original;
            }
        }
        if (lane == 0) {
            fps_idx[static_cast<int64_t>(b) * M + round_idx] = static_cast<int64_t>(best_original);
        }
    }
}

template <int D, int THREADS>
__global__ void fps_exact_bucketed_refresh_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ fps_idx,
    float* __restrict__ leaf_nearest_dist_sq,
    int* __restrict__ leaf_nearest_anchor,
    const int* __restrict__ bucket_leaf_begin,
    const int* __restrict__ bucket_leaf_end,
    float* __restrict__ bucket_max_dist_sq,
    int* __restrict__ bucket_max_leaf,
    const float* __restrict__ bucket_aabb,
    int* __restrict__ refreshed_counts,
    int* __restrict__ skipped_counts,
    int B,
    int N,
    int M,
    int bucket_count,
    int round_idx,
    bool enable_pruning,
    int count_stride
) {
    const int b = blockIdx.y;
    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || bucket >= bucket_count || round_idx >= M) {
        return;
    }

    const int64_t bucket_idx = static_cast<int64_t>(b) * bucket_count + bucket;
    const int64_t point_base = static_cast<int64_t>(b) * N * D;
    const int64_t leaf_base = static_cast<int64_t>(b) * N;
    const int anchor_idx = static_cast<int>(fps_idx[static_cast<int64_t>(b) * M + round_idx]);

    if (enable_pruning) {
        const float* q = points + point_base + static_cast<int64_t>(anchor_idx) * D;
        const float* aabb = bucket_aabb + bucket_idx * 2 * D;
        const float lower = min_distance_sq_to_aabb<D>(q, aabb);
        if (lower >= bucket_max_dist_sq[bucket_idx]) {
            if (tid == 0) {
                const int64_t count_idx = (count_stride == 0)
                    ? static_cast<int64_t>(b)
                    : static_cast<int64_t>(b) * count_stride + round_idx;
                atomicAdd(skipped_counts + count_idx, 1);
            }
            return;
        }
    }

    const int leaf_begin = bucket_leaf_begin[bucket_idx];
    const int leaf_end = bucket_leaf_end[bucket_idx];
    float local_best = -1.0f;
    int local_leaf = leaf_begin;
    int local_original = N;

    for (int leaf = leaf_begin + tid; leaf < leaf_end; leaf += THREADS) {
        const int original = static_cast<int>(sorted_indices[leaf_base + leaf]);
        float current = leaf_nearest_dist_sq[leaf_base + leaf];
        const float dist = point_distance_sq<D>(points, point_base, original, anchor_idx);
        if (dist < current) {
            current = dist;
            leaf_nearest_dist_sq[leaf_base + leaf] = current;
            leaf_nearest_anchor[leaf_base + leaf] = round_idx;
        }
        if (current > local_best || (current == local_best && original < local_original)) {
            local_best = current;
            local_leaf = leaf;
            local_original = original;
        }
    }

    __shared__ float shared_dist[THREADS];
    __shared__ int shared_leaf[THREADS];
    __shared__ int shared_original[THREADS];
    shared_dist[tid] = local_best;
    shared_leaf[tid] = local_leaf;
    shared_original[tid] = local_original;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_dist = shared_dist[tid + stride];
            const int other_original = shared_original[tid + stride];
            if (other_dist > shared_dist[tid]
                    || (other_dist == shared_dist[tid] && other_original < shared_original[tid])) {
                shared_dist[tid] = other_dist;
                shared_leaf[tid] = shared_leaf[tid + stride];
                shared_original[tid] = other_original;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        bucket_max_dist_sq[bucket_idx] = shared_dist[0];
        bucket_max_leaf[bucket_idx] = shared_leaf[0];
        const int64_t count_idx = (count_stride == 0)
            ? static_cast<int64_t>(b)
            : static_cast<int64_t>(b) * count_stride + round_idx;
        atomicAdd(refreshed_counts + count_idx, 1);
    }
}

template <int D>
__device__ inline float original_point_distance_sq(
    const float* __restrict__ points,
    int64_t point_base,
    int i,
    int j
) {
    float dist = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float delta = points[point_base + static_cast<int64_t>(i) * D + d]
            - points[point_base + static_cast<int64_t>(j) * D + d];
        dist += delta * delta;
    }
    return dist;
}

__device__ inline int fps_walk_max_leaf_from_mem(
    const float* __restrict__ sample_node_max,
    const int* __restrict__ left_child_mem,
    const int* __restrict__ right_child_mem,
    const int* __restrict__ mem_to_leaf,
    int root_mem
) {
    int mem_idx = root_mem;
    while (left_child_mem[mem_idx] != -1) {
        const int left = left_child_mem[mem_idx];
        const int right = right_child_mem[mem_idx];
        if (right != -1 && sample_node_max[right] > sample_node_max[left]) {
            mem_idx = right;
        } else {
            mem_idx = left;
        }
    }
    return mem_to_leaf[mem_idx];
}







template <int D>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_approx_bucketed_impl(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    int refresh_interval,
    int candidates_per_round,
    int anchors_per_round,
    float alpha,
    bool use_graph
) {
    (void)left_child_mem;
    (void)right_child_mem;
    (void)mem_to_leaf;
    (void)node_aabbs;
    (void)num_real_nodes;
    (void)refresh_interval;

    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    TORCH_CHECK(bucket_size >= 1, "fps_approx_bucketed: bucket_size must be >= 1");
    TORCH_CHECK(refresh_interval >= 1, "fps_approx_bucketed: refresh_interval must be >= 1");
    TORCH_CHECK(candidates_per_round >= 1, "fps_approx_bucketed: candidates_per_round must be >= 1");
    TORCH_CHECK(anchors_per_round >= 1 && anchors_per_round <= 8, "fps_approx_bucketed: anchors_per_round must be in [1, 8]");
    TORCH_CHECK(candidates_per_round <= 32, "fps_approx_bucketed: candidates_per_round must be <= 32");
    TORCH_CHECK(candidates_per_round >= anchors_per_round, "fps_approx_bucketed: candidates_per_round must be >= anchors_per_round");
    TORCH_CHECK(alpha >= 0.0f, "fps_approx_bucketed: alpha must be nonnegative");

    const int target_bucket_count = (N + bucket_size - 1) / bucket_size;
    int bucket_level = leaf_level;
    for (int level = 0; level <= leaf_level; ++level) {
        if (implicit_bvh::tree::real_nodes_at_level(N, level) >= target_bucket_count) {
            bucket_level = level;
            break;
        }
    }
    const int bucket_count = implicit_bvh::tree::real_nodes_at_level(N, bucket_level);
    const int min_bucket_size = N / bucket_count;
    const int max_bucket_size = (N + bucket_count - 1) / bucket_count;
    const int max_commit_per_iteration = min(anchors_per_round, bucket_count);
    const int max_iterations = (M <= 1) ? 0 : ((M - 1 + max_commit_per_iteration - 1) / max_commit_per_iteration);
    const int bucket_round_capacity = max(1, bucket_count / anchors_per_round);
    const int effective_refresh_interval = min(refresh_interval, bucket_round_capacity);
    const int dirty_refresh_interval = max(1, effective_refresh_interval / 2);

    // === CUDA GRAPH PATH ===
    if (use_graph) {
        constexpr int _threads = 256;
        constexpr int _max_candidates = 32;
        constexpr int _max_r = 8;
        const int _top_risk_buckets = min(candidates_per_round, max(anchors_per_round, anchors_per_round * 2));
        const dim3 _bucket_blocks(bucket_count, B);
        const int64_t _total_pts = static_cast<int64_t>(B) * N;
        const int _scatter_blocks = static_cast<int>((_total_pts + _threads - 1) / _threads);

        const BucketQueueWorkspaceKey _key{
            B, N, D, M, bucket_count,
            effective_refresh_interval, dirty_refresh_interval,
            max_iterations, max_commit_per_iteration,
            candidates_per_round, anchors_per_round, _top_risk_buckets, _threads
        };

        std::shared_ptr<BucketQueueWorkspace> _ws;
        {
            std::lock_guard<std::mutex> _lock(s_bucket_queue_workspace_mutex);
            auto _it = s_bucket_queue_workspace_cache.find(_key);
            if (_it != s_bucket_queue_workspace_cache.end()) {
                _ws = _it->second;
            } else {
                _ws = std::make_shared<BucketQueueWorkspace>();
                _ws->ws_points          = torch::empty({B, N, D}, points.options());
                _ws->ws_sorted_indices  = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt64));
                _ws->ws_seed_indices    = torch::empty({B}, seed_indices.options().dtype(torch::kInt64));
                _ws->fps_idx            = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt64));
                _ws->nearest_anchor     = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
                _ws->nearest_dist_sq    = torch::empty({B, N}, points.options());
                _ws->leaf_nearest_anchor  = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
                _ws->leaf_nearest_dist_sq = torch::empty({B, N}, points.options());
                _ws->selected_flags     = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
                _ws->selected_counts    = torch::empty({B}, seed_indices.options().dtype(torch::kInt32));
                _ws->bucket_leaf_begin  = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
                _ws->bucket_leaf_end    = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
                _ws->bucket_last_applied = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
                _ws->bucket_max_dist_sq = torch::empty({B, bucket_count}, points.options());
                _ws->bucket_max_leaf    = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
                _ws->bucket_aabb        = torch::empty({B, bucket_count, 2 * D}, points.options());
                _ws->dirty_flags        = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
                _ws->mode_flags         = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                _ws->visited_leaf_counts = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                _ws->committed_counts   = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                _ws->candidate_counts   = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                _ws->rejected_counts    = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                _ws->refresh_flags      = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt32));
                // Dedicated non-null stream for graph capture/replay
                C10_CUDA_CHECK(cudaStreamCreateWithFlags(&_ws->ws_stream, cudaStreamNonBlocking));
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&_ws->ws_input_ready, cudaEventDisableTiming));
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&_ws->ws_output_ready, cudaEventDisableTiming));
                s_bucket_queue_workspace_cache[_key] = _ws;
            }
        }

        // Input copies on the main (ATen) stream, which may be the null/legacy-default stream.
        // Using raw cudaMemcpyAsync avoids any PyTorch stream bookkeeping that could interfere
        // with capture mode on the dedicated workspace stream below.
        cudaStream_t _main_stream = at::cuda::getCurrentCUDAStream().stream();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            _ws->ws_points.data_ptr(),
            points.data_ptr(),
            static_cast<size_t>(B) * N * D * sizeof(float),
            cudaMemcpyDeviceToDevice, _main_stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            _ws->ws_sorted_indices.data_ptr(),
            sorted_indices.data_ptr(),
            static_cast<size_t>(B) * N * sizeof(int64_t),
            cudaMemcpyDeviceToDevice, _main_stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            _ws->ws_seed_indices.data_ptr(),
            seed_indices.data_ptr(),
            static_cast<size_t>(B) * sizeof(int64_t),
            cudaMemcpyDeviceToDevice, _main_stream));

        // Signal when input copies are done on main stream; make ws_stream wait.
        C10_CUDA_CHECK(cudaEventRecord(_ws->ws_input_ready, _main_stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(_ws->ws_stream, _ws->ws_input_ready, 0));

        if (!_ws->graph_exec) {
            // Drain ws_stream (the event-wait above) before capture so the stream is idle.
            C10_CUDA_CHECK(cudaStreamSynchronize(_ws->ws_stream));
            // First call for this config: capture the entire loop as a CUDA graph.
            C10_CUDA_CHECK(cudaStreamBeginCapture(_ws->ws_stream, cudaStreamCaptureModeRelaxed));

            // Zero-initialize tensors that must start as zero on every call.
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->selected_flags.data_ptr(),      0, static_cast<size_t>(B) * N            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->dirty_flags.data_ptr(),         0, static_cast<size_t>(B) * bucket_count  * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->mode_flags.data_ptr(),          0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->visited_leaf_counts.data_ptr(), 0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->committed_counts.data_ptr(),    0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->candidate_counts.data_ptr(),    0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->rejected_counts.data_ptr(),     0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->refresh_flags.data_ptr(),       0, static_cast<size_t>(B) * M            * sizeof(int), _ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(_ws->fps_idx.data_ptr(),            0, static_cast<size_t>(B) * M            * sizeof(int64_t), _ws->ws_stream));

            fps_bucket_queue_init_kernel<D, _threads><<<_bucket_blocks, _threads, 0, _ws->ws_stream>>>(
                _ws->ws_points.data_ptr<float>(),
                _ws->ws_seed_indices.data_ptr<int64_t>(),
                _ws->ws_sorted_indices.data_ptr<int64_t>(),
                _ws->fps_idx.data_ptr<int64_t>(),
                _ws->selected_flags.data_ptr<int>(),
                _ws->selected_counts.data_ptr<int>(),
                _ws->leaf_nearest_dist_sq.data_ptr<float>(),
                _ws->leaf_nearest_anchor.data_ptr<int>(),
                _ws->bucket_leaf_begin.data_ptr<int>(),
                _ws->bucket_leaf_end.data_ptr<int>(),
                _ws->bucket_last_applied.data_ptr<int>(),
                _ws->bucket_max_dist_sq.data_ptr<float>(),
                _ws->bucket_max_leaf.data_ptr<int>(),
                _ws->bucket_aabb.data_ptr<float>(),
                B, N, M, leaf_level, bucket_level, bucket_count
            );

            for (int _iter = 0; _iter < max_iterations; ++_iter) {
                const int _stat_slot = 1 + _iter * max_commit_per_iteration;
                const int _committed_start = 1 + _iter * max_commit_per_iteration;
                fps_bucket_queue_select_kernel<_threads, _max_candidates, _max_r><<<B, _threads, 0, _ws->ws_stream>>>(
                    _ws->ws_sorted_indices.data_ptr<int64_t>(),
                    _ws->bucket_max_dist_sq.data_ptr<float>(),
                    _ws->bucket_max_leaf.data_ptr<int>(),
                    _ws->fps_idx.data_ptr<int64_t>(),
                    _ws->selected_flags.data_ptr<int>(),
                    _ws->selected_counts.data_ptr<int>(),
                    _ws->mode_flags.data_ptr<int>(),
                    _ws->visited_leaf_counts.data_ptr<int>(),
                    _ws->committed_counts.data_ptr<int>(),
                    _ws->candidate_counts.data_ptr<int>(),
                    _ws->rejected_counts.data_ptr<int>(),
                    _ws->dirty_flags.data_ptr<int>(),
                    B, N, M, bucket_count,
                    candidates_per_round, anchors_per_round, _top_risk_buckets,
                    alpha, _stat_slot
                );

                fps_mark_dirty_aabb_kernel<D, _threads><<<_bucket_blocks, _threads, 0, _ws->ws_stream>>>(
                    _ws->ws_points.data_ptr<float>(),
                    _ws->fps_idx.data_ptr<int64_t>(),
                    _ws->selected_flags.data_ptr<int>(),
                    _ws->bucket_aabb.data_ptr<float>(),
                    _ws->bucket_max_dist_sq.data_ptr<float>(),
                    _ws->dirty_flags.data_ptr<int>(),
                    B, N, M, bucket_count,
                    _committed_start, max_commit_per_iteration
                );

                const bool _force_refresh = ((_iter + 1) % effective_refresh_interval) == 0
                    || _iter == max_iterations - 1;
                const bool _dirty_refresh = ((_iter + 1) % dirty_refresh_interval) == 0;
                if (_force_refresh || _dirty_refresh) {
                    fps_bucket_queue_refresh_kernel<D, _threads><<<_bucket_blocks, _threads, 0, _ws->ws_stream>>>(
                        _ws->ws_points.data_ptr<float>(),
                        _ws->ws_sorted_indices.data_ptr<int64_t>(),
                        _ws->fps_idx.data_ptr<int64_t>(),
                        _ws->selected_flags.data_ptr<int>(),
                        _ws->selected_counts.data_ptr<int>(),
                        _ws->leaf_nearest_dist_sq.data_ptr<float>(),
                        _ws->leaf_nearest_anchor.data_ptr<int>(),
                        _ws->bucket_leaf_begin.data_ptr<int>(),
                        _ws->bucket_leaf_end.data_ptr<int>(),
                        _ws->bucket_last_applied.data_ptr<int>(),
                        _ws->bucket_max_dist_sq.data_ptr<float>(),
                        _ws->bucket_max_leaf.data_ptr<int>(),
                        _ws->dirty_flags.data_ptr<int>(),
                        _ws->refresh_flags.data_ptr<int>(),
                        B, N, M, bucket_count, _force_refresh, _stat_slot
                    );
                }
            }

            fps_scatter_leaf_state_kernel<<<_scatter_blocks, _threads, 0, _ws->ws_stream>>>(
                _ws->ws_sorted_indices.data_ptr<int64_t>(),
                _ws->leaf_nearest_anchor.data_ptr<int>(),
                _ws->leaf_nearest_dist_sq.data_ptr<float>(),
                _ws->nearest_anchor.data_ptr<int>(),
                _ws->nearest_dist_sq.data_ptr<float>(),
                B, N
            );

            C10_CUDA_CHECK(cudaStreamEndCapture(_ws->ws_stream, &_ws->graph));
            C10_CUDA_CHECK(cudaGraphInstantiate(&_ws->graph_exec, _ws->graph, nullptr, nullptr, 0));
        }

        C10_CUDA_CHECK(cudaGraphLaunch(_ws->graph_exec, _ws->ws_stream));
        // Signal graph done; make main stream wait before reading outputs or proceeding.
        C10_CUDA_CHECK(cudaEventRecord(_ws->ws_output_ready, _ws->ws_stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(_main_stream, _ws->ws_output_ready, 0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Build bucket_info on the CPU (outside the graph) and copy to device.
        auto _bucket_info = torch::empty({16}, seed_indices.options().dtype(torch::kInt32));
        auto _bucket_info_cpu = torch::empty({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        auto* _info = _bucket_info_cpu.data_ptr<int>();
        _info[0]  = bucket_level;
        _info[1]  = bucket_count;
        _info[2]  = min_bucket_size;
        _info[3]  = max_bucket_size;
        _info[4]  = bucket_size;
        _info[5]  = refresh_interval;
        _info[6]  = candidates_per_round;
        _info[7]  = anchors_per_round;
        _info[8]  = max_iterations;
        _info[9]  = bucket_count * ((max_iterations + effective_refresh_interval - 1) / effective_refresh_interval);
        _info[10] = 0;
        _info[11] = _top_risk_buckets;
        _info[12] = dirty_refresh_interval;
        _info[13] = 0;
        _info[14] = 0;
        _info[15] = 2;
        _bucket_info.copy_(_bucket_info_cpu, false);

        // Clone workspace outputs so each caller gets independent tensors.
        return std::make_tuple(
            _ws->fps_idx.clone(),
            _ws->nearest_anchor.clone(),
            _ws->nearest_dist_sq.clone(),
            _ws->mode_flags.clone(),
            _ws->visited_leaf_counts.clone(),
            _ws->committed_counts.clone(),
            _ws->candidate_counts.clone(),
            _ws->rejected_counts.clone(),
            _ws->refresh_flags.clone(),
            _bucket_info
        );
    }
    // === END CUDA GRAPH PATH ===

    auto fps_idx = torch::empty({B, M}, seed_indices.options().dtype(torch::kInt64));
    auto nearest_anchor = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
    auto nearest_dist_sq = torch::empty({B, N}, points.options());
    auto leaf_nearest_anchor = torch::empty({B, N}, seed_indices.options().dtype(torch::kInt32));
    auto leaf_nearest_dist_sq = torch::empty({B, N}, points.options());
    auto selected_flags = torch::zeros({B, N}, seed_indices.options().dtype(torch::kInt32));
    auto selected_counts = torch::empty({B}, seed_indices.options().dtype(torch::kInt32));

    auto bucket_leaf_begin = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
    auto bucket_leaf_end = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
    auto bucket_last_applied = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
    auto bucket_max_dist_sq = torch::empty({B, bucket_count}, points.options());
    auto bucket_max_leaf = torch::empty({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));
    auto bucket_aabb = torch::empty({B, bucket_count, 2 * D}, points.options());
    auto dirty_flags = torch::zeros({B, bucket_count}, seed_indices.options().dtype(torch::kInt32));

    auto mode_flags = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));
    auto visited_leaf_counts = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));
    auto committed_counts = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));
    auto candidate_counts = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));
    auto rejected_counts = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));
    auto refresh_flags = torch::zeros({B, M}, seed_indices.options().dtype(torch::kInt32));

    constexpr int threads = 256;
    constexpr int max_candidates = 32;
    constexpr int max_r = 8;
    const int top_risk_buckets = min(candidates_per_round, max(anchors_per_round, anchors_per_round * 2));
    const dim3 bucket_blocks(bucket_count, B);
    fps_bucket_queue_init_kernel<D, threads><<<bucket_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        seed_indices.data_ptr<int64_t>(),
        sorted_indices.data_ptr<int64_t>(),
        fps_idx.data_ptr<int64_t>(),
        selected_flags.data_ptr<int>(),
        selected_counts.data_ptr<int>(),
        leaf_nearest_dist_sq.data_ptr<float>(),
        leaf_nearest_anchor.data_ptr<int>(),
        bucket_leaf_begin.data_ptr<int>(),
        bucket_leaf_end.data_ptr<int>(),
        bucket_last_applied.data_ptr<int>(),
        bucket_max_dist_sq.data_ptr<float>(),
        bucket_max_leaf.data_ptr<int>(),
        bucket_aabb.data_ptr<float>(),
        B,
        N,
        M,
        leaf_level,
        bucket_level,
        bucket_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int iter = 0; iter < max_iterations; ++iter) {
        const int stat_slot = 1 + iter * max_commit_per_iteration;
        const int committed_start = 1 + iter * max_commit_per_iteration;
        fps_bucket_queue_select_kernel<threads, max_candidates, max_r><<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sorted_indices.data_ptr<int64_t>(),
            bucket_max_dist_sq.data_ptr<float>(),
            bucket_max_leaf.data_ptr<int>(),
            fps_idx.data_ptr<int64_t>(),
            selected_flags.data_ptr<int>(),
            selected_counts.data_ptr<int>(),
            mode_flags.data_ptr<int>(),
            visited_leaf_counts.data_ptr<int>(),
            committed_counts.data_ptr<int>(),
            candidate_counts.data_ptr<int>(),
            rejected_counts.data_ptr<int>(),
            dirty_flags.data_ptr<int>(),
            B,
            N,
            M,
            bucket_count,
            candidates_per_round,
            anchors_per_round,
            top_risk_buckets,
            alpha,
            stat_slot
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fps_mark_dirty_aabb_kernel<D, threads><<<bucket_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<float>(),
            fps_idx.data_ptr<int64_t>(),
            selected_flags.data_ptr<int>(),
            bucket_aabb.data_ptr<float>(),
            bucket_max_dist_sq.data_ptr<float>(),
            dirty_flags.data_ptr<int>(),
            B, N, M, bucket_count,
            committed_start, max_commit_per_iteration
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const bool force_refresh = ((iter + 1) % effective_refresh_interval) == 0
            || iter == max_iterations - 1;
        const bool dirty_refresh = ((iter + 1) % dirty_refresh_interval) == 0;
        if (force_refresh || dirty_refresh) {
            fps_bucket_queue_refresh_kernel<D, threads><<<bucket_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                points.data_ptr<float>(),
                sorted_indices.data_ptr<int64_t>(),
                fps_idx.data_ptr<int64_t>(),
                selected_flags.data_ptr<int>(),
                selected_counts.data_ptr<int>(),
                leaf_nearest_dist_sq.data_ptr<float>(),
                leaf_nearest_anchor.data_ptr<int>(),
                bucket_leaf_begin.data_ptr<int>(),
                bucket_leaf_end.data_ptr<int>(),
                bucket_last_applied.data_ptr<int>(),
                bucket_max_dist_sq.data_ptr<float>(),
                bucket_max_leaf.data_ptr<int>(),
                dirty_flags.data_ptr<int>(),
                refresh_flags.data_ptr<int>(),
                B,
                N,
                M,
                bucket_count,
                force_refresh,
                stat_slot
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    const int64_t total = static_cast<int64_t>(B) * N;
    const int scatter_blocks = static_cast<int>((total + threads - 1) / threads);
    fps_scatter_leaf_state_kernel<<<scatter_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        sorted_indices.data_ptr<int64_t>(),
        leaf_nearest_anchor.data_ptr<int>(),
        leaf_nearest_dist_sq.data_ptr<float>(),
        nearest_anchor.data_ptr<int>(),
        nearest_dist_sq.data_ptr<float>(),
        B,
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto bucket_info = torch::empty({16}, seed_indices.options().dtype(torch::kInt32));
    auto bucket_info_cpu = torch::empty({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* info = bucket_info_cpu.data_ptr<int>();
    info[0] = bucket_level;
    info[1] = bucket_count;
    info[2] = min_bucket_size;
    info[3] = max_bucket_size;
    info[4] = bucket_size;
    info[5] = refresh_interval;
    info[6] = candidates_per_round;
    info[7] = anchors_per_round;
    info[8] = max_iterations;
    info[9] = bucket_count * ((max_iterations + effective_refresh_interval - 1) / effective_refresh_interval);
    info[10] = 0;
    info[11] = top_risk_buckets;
    info[12] = dirty_refresh_interval;
    info[13] = 0;
    info[14] = 0;
    info[15] = 2;
    bucket_info.copy_(bucket_info_cpu, false);

    return std::make_tuple(
        fps_idx,
        nearest_anchor,
        nearest_dist_sq,
        mode_flags,
        visited_leaf_counts,
        committed_counts,
        candidate_counts,
        rejected_counts,
        refresh_flags,
        bucket_info
    );
}

// ---------------------------------------------------------------------------
// Exact bucketed graph route and workspace cache
// ---------------------------------------------------------------------------

struct ExactBucketedWorkspaceKey {
    int B, N, D, M;
    int bucket_count;
    int bucket_size;
    int enable_pruning;
    int threads_per_block;
    int device_index;

    bool operator==(const ExactBucketedWorkspaceKey& o) const noexcept {
        return B==o.B && N==o.N && D==o.D && M==o.M
            && bucket_count==o.bucket_count
            && bucket_size==o.bucket_size
            && enable_pruning==o.enable_pruning
            && threads_per_block==o.threads_per_block
            && device_index==o.device_index;
    }
};

struct ExactBucketedWorkspaceKeyHash {
    std::size_t operator()(const ExactBucketedWorkspaceKey& k) const noexcept {
        std::size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(k.B); mix(k.N); mix(k.D); mix(k.M);
        mix(k.bucket_count); mix(k.bucket_size);
        mix(k.enable_pruning); mix(k.threads_per_block);
        mix(k.device_index);
        return h;
    }
};

struct ExactBucketedWorkspace {
    torch::Tensor ws_points, ws_sorted_indices, ws_seed_indices;
    torch::Tensor fps_idx, nearest_anchor, nearest_dist_sq;
    torch::Tensor leaf_nearest_anchor, leaf_nearest_dist_sq;
    torch::Tensor bucket_leaf_begin, bucket_leaf_end;
    torch::Tensor bucket_max_dist_sq, bucket_max_leaf, bucket_aabb;
    torch::Tensor refreshed_counts, skipped_counts;

    cudaStream_t ws_stream = nullptr;
    cudaEvent_t ws_input_ready = nullptr;
    cudaEvent_t ws_output_ready = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    ~ExactBucketedWorkspace() {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        if (ws_output_ready) { cudaEventDestroy(ws_output_ready); ws_output_ready = nullptr; }
        if (ws_input_ready) { cudaEventDestroy(ws_input_ready); ws_input_ready = nullptr; }
        if (ws_stream) { cudaStreamDestroy(ws_stream); ws_stream = nullptr; }
    }
};

static std::unordered_map<
    ExactBucketedWorkspaceKey,
    std::shared_ptr<ExactBucketedWorkspace>,
    ExactBucketedWorkspaceKeyHash
> s_exact_bucketed_workspace_cache;
static std::mutex s_exact_bucketed_workspace_mutex;

template <int D>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_exact_bucketed_impl(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    bool use_graph,
    bool enable_pruning
) {
    (void)left_child_mem;
    (void)right_child_mem;
    (void)mem_to_leaf;
    (void)node_aabbs;
    (void)num_real_nodes;
    (void)use_graph;

    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    TORCH_CHECK(bucket_size >= 1, "fps_exact_bucketed: bucket_size must be >= 1");

    const int target_bucket_count = (N + bucket_size - 1) / bucket_size;
    int bucket_level = leaf_level;
    for (int level = 0; level <= leaf_level; ++level) {
        if (implicit_bvh::tree::real_nodes_at_level(N, level) >= target_bucket_count) {
            bucket_level = level;
            break;
        }
    }
    const int bucket_count = implicit_bvh::tree::real_nodes_at_level(N, bucket_level);
    const int min_bucket_size = N / bucket_count;
    const int max_bucket_size = (N + bucket_count - 1) / bucket_count;

    auto idx_options = seed_indices.options().dtype(torch::kInt64);
    auto i32_options = seed_indices.options().dtype(torch::kInt32);
    auto f_options = points.options();

    constexpr int threads = 256;
    const dim3 bucket_blocks(bucket_count, B);
    const int64_t total = static_cast<int64_t>(B) * N;
    const int scatter_blocks = static_cast<int>((total + threads - 1) / threads);

    if (use_graph) {
        const ExactBucketedWorkspaceKey key{
            B, N, D, M, bucket_count, bucket_size, enable_pruning ? 1 : 0,
            threads, points.get_device()
        };

        std::shared_ptr<ExactBucketedWorkspace> ws;
        {
            std::lock_guard<std::mutex> lock(s_exact_bucketed_workspace_mutex);
            auto it = s_exact_bucketed_workspace_cache.find(key);
            if (it != s_exact_bucketed_workspace_cache.end()) {
                ws = it->second;
            } else {
                ws = std::make_shared<ExactBucketedWorkspace>();
                ws->ws_points = torch::empty({B, N, D}, f_options);
                ws->ws_sorted_indices = torch::empty({B, N}, idx_options);
                ws->ws_seed_indices = torch::empty({B}, idx_options);
                ws->fps_idx = torch::empty({B, M}, idx_options);
                ws->nearest_anchor = torch::empty({B, N}, i32_options);
                ws->nearest_dist_sq = torch::empty({B, N}, f_options);
                ws->leaf_nearest_anchor = torch::empty({B, N}, i32_options);
                ws->leaf_nearest_dist_sq = torch::empty({B, N}, f_options);
                ws->bucket_leaf_begin = torch::empty({B, bucket_count}, i32_options);
                ws->bucket_leaf_end = torch::empty({B, bucket_count}, i32_options);
                ws->bucket_max_dist_sq = torch::empty({B, bucket_count}, f_options);
                ws->bucket_max_leaf = torch::empty({B, bucket_count}, i32_options);
                ws->bucket_aabb = torch::empty({B, bucket_count, 2 * D}, f_options);
                ws->refreshed_counts = torch::empty({B}, i32_options);
                ws->skipped_counts = torch::empty({B}, i32_options);
                C10_CUDA_CHECK(cudaStreamCreateWithFlags(&ws->ws_stream, cudaStreamNonBlocking));
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&ws->ws_input_ready, cudaEventDisableTiming));
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&ws->ws_output_ready, cudaEventDisableTiming));
                s_exact_bucketed_workspace_cache[key] = ws;
            }
        }

        cudaStream_t main_stream = at::cuda::getCurrentCUDAStream().stream();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ws->ws_points.data_ptr(),
            points.data_ptr(),
            static_cast<size_t>(B) * N * D * sizeof(float),
            cudaMemcpyDeviceToDevice, main_stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ws->ws_sorted_indices.data_ptr(),
            sorted_indices.data_ptr(),
            static_cast<size_t>(B) * N * sizeof(int64_t),
            cudaMemcpyDeviceToDevice, main_stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            ws->ws_seed_indices.data_ptr(),
            seed_indices.data_ptr(),
            static_cast<size_t>(B) * sizeof(int64_t),
            cudaMemcpyDeviceToDevice, main_stream));
        C10_CUDA_CHECK(cudaEventRecord(ws->ws_input_ready, main_stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(ws->ws_stream, ws->ws_input_ready, 0));

        if (!ws->graph_exec) {
            C10_CUDA_CHECK(cudaStreamSynchronize(ws->ws_stream));
            C10_CUDA_CHECK(cudaStreamBeginCapture(ws->ws_stream, cudaStreamCaptureModeRelaxed));

            C10_CUDA_CHECK(cudaMemsetAsync(ws->refreshed_counts.data_ptr(), 0, static_cast<size_t>(B) * sizeof(int), ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(ws->skipped_counts.data_ptr(), 0, static_cast<size_t>(B) * sizeof(int), ws->ws_stream));
            C10_CUDA_CHECK(cudaMemsetAsync(ws->fps_idx.data_ptr(), 0, static_cast<size_t>(B) * M * sizeof(int64_t), ws->ws_stream));

            fps_exact_bucketed_init_kernel<D, threads><<<bucket_blocks, threads, 0, ws->ws_stream>>>(
                ws->ws_points.data_ptr<float>(),
                ws->ws_seed_indices.data_ptr<int64_t>(),
                ws->ws_sorted_indices.data_ptr<int64_t>(),
                ws->fps_idx.data_ptr<int64_t>(),
                ws->leaf_nearest_dist_sq.data_ptr<float>(),
                ws->leaf_nearest_anchor.data_ptr<int>(),
                ws->bucket_leaf_begin.data_ptr<int>(),
                ws->bucket_leaf_end.data_ptr<int>(),
                ws->bucket_max_dist_sq.data_ptr<float>(),
                ws->bucket_max_leaf.data_ptr<int>(),
                ws->bucket_aabb.data_ptr<float>(),
                ws->refreshed_counts.data_ptr<int>(),
                B, N, M, leaf_level, bucket_level, bucket_count, 0
            );

            for (int round_idx = 1; round_idx < M; ++round_idx) {
                fps_exact_bucketed_select_kernel<threads><<<B, threads, 0, ws->ws_stream>>>(
                    ws->ws_sorted_indices.data_ptr<int64_t>(),
                    ws->bucket_max_dist_sq.data_ptr<float>(),
                    ws->bucket_max_leaf.data_ptr<int>(),
                    ws->fps_idx.data_ptr<int64_t>(),
                    B, N, M, bucket_count, round_idx
                );

                fps_exact_bucketed_refresh_kernel<D, threads><<<bucket_blocks, threads, 0, ws->ws_stream>>>(
                    ws->ws_points.data_ptr<float>(),
                    ws->ws_sorted_indices.data_ptr<int64_t>(),
                    ws->fps_idx.data_ptr<int64_t>(),
                    ws->leaf_nearest_dist_sq.data_ptr<float>(),
                    ws->leaf_nearest_anchor.data_ptr<int>(),
                    ws->bucket_leaf_begin.data_ptr<int>(),
                    ws->bucket_leaf_end.data_ptr<int>(),
                    ws->bucket_max_dist_sq.data_ptr<float>(),
                    ws->bucket_max_leaf.data_ptr<int>(),
                    ws->bucket_aabb.data_ptr<float>(),
                    ws->refreshed_counts.data_ptr<int>(),
                    ws->skipped_counts.data_ptr<int>(),
                    B, N, M, bucket_count, round_idx, enable_pruning, 0
                );
            }

            fps_scatter_leaf_state_kernel<<<scatter_blocks, threads, 0, ws->ws_stream>>>(
                ws->ws_sorted_indices.data_ptr<int64_t>(),
                ws->leaf_nearest_anchor.data_ptr<int>(),
                ws->leaf_nearest_dist_sq.data_ptr<float>(),
                ws->nearest_anchor.data_ptr<int>(),
                ws->nearest_dist_sq.data_ptr<float>(),
                B,
                N
            );

            C10_CUDA_CHECK(cudaStreamEndCapture(ws->ws_stream, &ws->graph));
            C10_CUDA_CHECK(cudaGraphInstantiate(&ws->graph_exec, ws->graph, nullptr, nullptr, 0));
        }

        C10_CUDA_CHECK(cudaGraphLaunch(ws->graph_exec, ws->ws_stream));
        C10_CUDA_CHECK(cudaEventRecord(ws->ws_output_ready, ws->ws_stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(main_stream, ws->ws_output_ready, 0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto bucket_info = torch::empty({16}, i32_options);
        auto bucket_info_cpu = torch::empty({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        auto* info = bucket_info_cpu.data_ptr<int>();
        info[0] = bucket_level;
        info[1] = bucket_count;
        info[2] = min_bucket_size;
        info[3] = max_bucket_size;
        info[4] = bucket_size;
        info[5] = enable_pruning ? 1 : 0;
        info[6] = 1;
        info[7] = 1;
        info[8] = M > 0 ? M - 1 : 0;
        info[9] = 1;
        for (int i = 10; i < 16; ++i) {
            info[i] = 0;
        }
        bucket_info.copy_(bucket_info_cpu, false);

        return std::make_tuple(
            ws->fps_idx.clone(),
            ws->nearest_anchor.clone(),
            ws->nearest_dist_sq.clone(),
            ws->refreshed_counts.clone(),
            ws->skipped_counts.clone(),
            bucket_info
        );
    }

    auto fps_idx = torch::empty({B, M}, idx_options);
    auto nearest_anchor = torch::empty({B, N}, i32_options);
    auto nearest_dist_sq = torch::empty({B, N}, f_options);
    auto leaf_nearest_anchor = torch::empty({B, N}, i32_options);
    auto leaf_nearest_dist_sq = torch::empty({B, N}, f_options);
    auto bucket_leaf_begin = torch::empty({B, bucket_count}, i32_options);
    auto bucket_leaf_end = torch::empty({B, bucket_count}, i32_options);
    auto bucket_max_dist_sq = torch::empty({B, bucket_count}, f_options);
    auto bucket_max_leaf = torch::empty({B, bucket_count}, i32_options);
    auto bucket_aabb = torch::empty({B, bucket_count, 2 * D}, f_options);
    auto refreshed_counts = torch::zeros({B}, i32_options);
    auto skipped_counts = torch::zeros({B}, i32_options);

    fps_exact_bucketed_init_kernel<D, threads><<<bucket_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(),
        seed_indices.data_ptr<int64_t>(),
        sorted_indices.data_ptr<int64_t>(),
        fps_idx.data_ptr<int64_t>(),
        leaf_nearest_dist_sq.data_ptr<float>(),
        leaf_nearest_anchor.data_ptr<int>(),
        bucket_leaf_begin.data_ptr<int>(),
        bucket_leaf_end.data_ptr<int>(),
        bucket_max_dist_sq.data_ptr<float>(),
        bucket_max_leaf.data_ptr<int>(),
        bucket_aabb.data_ptr<float>(),
        refreshed_counts.data_ptr<int>(),
        B, N, M, leaf_level, bucket_level, bucket_count, 0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int round_idx = 1; round_idx < M; ++round_idx) {
        fps_exact_bucketed_select_kernel<threads><<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sorted_indices.data_ptr<int64_t>(),
            bucket_max_dist_sq.data_ptr<float>(),
            bucket_max_leaf.data_ptr<int>(),
            fps_idx.data_ptr<int64_t>(),
            B, N, M, bucket_count, round_idx
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fps_exact_bucketed_refresh_kernel<D, threads><<<bucket_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<float>(),
            sorted_indices.data_ptr<int64_t>(),
            fps_idx.data_ptr<int64_t>(),
            leaf_nearest_dist_sq.data_ptr<float>(),
            leaf_nearest_anchor.data_ptr<int>(),
            bucket_leaf_begin.data_ptr<int>(),
            bucket_leaf_end.data_ptr<int>(),
            bucket_max_dist_sq.data_ptr<float>(),
            bucket_max_leaf.data_ptr<int>(),
            bucket_aabb.data_ptr<float>(),
            refreshed_counts.data_ptr<int>(),
            skipped_counts.data_ptr<int>(),
            B, N, M, bucket_count, round_idx, enable_pruning, 0
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    fps_scatter_leaf_state_kernel<<<scatter_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        sorted_indices.data_ptr<int64_t>(),
        leaf_nearest_anchor.data_ptr<int>(),
        leaf_nearest_dist_sq.data_ptr<float>(),
        nearest_anchor.data_ptr<int>(),
        nearest_dist_sq.data_ptr<float>(),
        B,
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto bucket_info = torch::empty({16}, i32_options);
    auto bucket_info_cpu = torch::empty({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* info = bucket_info_cpu.data_ptr<int>();
    info[0] = bucket_level;
    info[1] = bucket_count;
    info[2] = min_bucket_size;
    info[3] = max_bucket_size;
    info[4] = bucket_size;
    info[5] = enable_pruning ? 1 : 0;
    info[6] = use_graph ? 1 : 0;
    info[7] = 0; // graph capture is intentionally not used by this exact fallback path
    info[8] = M > 0 ? M - 1 : 0;
    info[9] = 1; // route id: exact bucketed
    for (int i = 10; i < 16; ++i) {
        info[i] = 0;
    }
    bucket_info.copy_(bucket_info_cpu, false);

    return std::make_tuple(
        fps_idx,
        nearest_anchor,
        nearest_dist_sq,
        refreshed_counts,
        skipped_counts,
        bucket_info
    );
}




std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_approx_bucketed_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    int refresh_interval,
    int candidates_per_round,
    int anchors_per_round,
    double alpha,
    bool use_graph
) {
    TORCH_CHECK(points.is_cuda(), "fps_approx_bucketed: points must be a CUDA tensor");
    TORCH_CHECK(seed_indices.is_cuda(), "fps_approx_bucketed: seed_indices must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "fps_approx_bucketed: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(left_child_mem.is_cuda(), "fps_approx_bucketed: left_child_mem must be a CUDA tensor");
    TORCH_CHECK(right_child_mem.is_cuda(), "fps_approx_bucketed: right_child_mem must be a CUDA tensor");
    TORCH_CHECK(mem_to_leaf.is_cuda(), "fps_approx_bucketed: mem_to_leaf must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_cuda(), "fps_approx_bucketed: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "fps_approx_bucketed: points must be contiguous");
    TORCH_CHECK(seed_indices.is_contiguous(), "fps_approx_bucketed: seed_indices must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "fps_approx_bucketed: sorted_indices must be contiguous");
    TORCH_CHECK(left_child_mem.is_contiguous(), "fps_approx_bucketed: left_child_mem must be contiguous");
    TORCH_CHECK(right_child_mem.is_contiguous(), "fps_approx_bucketed: right_child_mem must be contiguous");
    TORCH_CHECK(mem_to_leaf.is_contiguous(), "fps_approx_bucketed: mem_to_leaf must be contiguous");
    TORCH_CHECK(node_aabbs.is_contiguous(), "fps_approx_bucketed: node_aabbs must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "fps_approx_bucketed: points must be float32");
    TORCH_CHECK(seed_indices.scalar_type() == torch::kInt64, "fps_approx_bucketed: seed_indices must be int64");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "fps_approx_bucketed: sorted_indices must be int64");
    TORCH_CHECK(left_child_mem.scalar_type() == torch::kInt32, "fps_approx_bucketed: left_child_mem must be int32");
    TORCH_CHECK(right_child_mem.scalar_type() == torch::kInt32, "fps_approx_bucketed: right_child_mem must be int32");
    TORCH_CHECK(mem_to_leaf.scalar_type() == torch::kInt32, "fps_approx_bucketed: mem_to_leaf must be int32");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "fps_approx_bucketed: node_aabbs must be float32");
    TORCH_CHECK(points.dim() == 3, "fps_approx_bucketed: points must have shape (B, N, D)");
    TORCH_CHECK(seed_indices.dim() == 1, "fps_approx_bucketed: seed_indices must have shape (B,)");
    TORCH_CHECK(sorted_indices.dim() == 2, "fps_approx_bucketed: sorted_indices must have shape (B, N)");
    TORCH_CHECK(node_aabbs.dim() == 3, "fps_approx_bucketed: node_aabbs must have shape (B, num_real_nodes, 2 * D)");
    TORCH_CHECK(left_child_mem.dim() == 1 && right_child_mem.dim() == 1 && mem_to_leaf.dim() == 1, "fps_approx_bucketed: child/leaf arrays must be 1D");
    TORCH_CHECK(points.size(0) >= 1, "fps_approx_bucketed: batch size must be at least 1");
    TORCH_CHECK(points.size(1) >= 1, "fps_approx_bucketed: points must contain at least one point per sample");
    TORCH_CHECK(points.size(2) == 2 || points.size(2) == 3, "fps_approx_bucketed: D must be 2 or 3");

    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int D = static_cast<int>(points.size(2));
    TORCH_CHECK(seed_indices.size(0) == B, "fps_approx_bucketed: seed_indices length must match batch size");
    TORCH_CHECK(sorted_indices.size(0) == B && sorted_indices.size(1) == N, "fps_approx_bucketed: sorted_indices shape must match points");
    TORCH_CHECK(node_aabbs.size(0) == B && node_aabbs.size(1) == num_real_nodes && node_aabbs.size(2) == 2 * D, "fps_approx_bucketed: node_aabbs shape mismatch");
    TORCH_CHECK(left_child_mem.size(0) == num_real_nodes, "fps_approx_bucketed: left_child_mem length must match num_real_nodes");
    TORCH_CHECK(right_child_mem.size(0) == num_real_nodes, "fps_approx_bucketed: right_child_mem length must match num_real_nodes");
    TORCH_CHECK(mem_to_leaf.size(0) == num_real_nodes, "fps_approx_bucketed: mem_to_leaf length must match num_real_nodes");
    TORCH_CHECK(leaf_level == implicit_bvh::tree::leaf_level(N), "fps_approx_bucketed: leaf_level must match N");
    TORCH_CHECK(M >= 1 && M <= N, "fps_approx_bucketed: target token count must be in [1, N]");

    c10::cuda::CUDAGuard device_guard(points.device());
    TORCH_CHECK(seed_indices.device() == points.device(), "fps_approx_bucketed: seed_indices and points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == points.device(), "fps_approx_bucketed: sorted_indices and points must be on the same device");
    TORCH_CHECK(left_child_mem.device() == points.device(), "fps_approx_bucketed: left_child_mem and points must be on the same device");
    TORCH_CHECK(right_child_mem.device() == points.device(), "fps_approx_bucketed: right_child_mem and points must be on the same device");
    TORCH_CHECK(mem_to_leaf.device() == points.device(), "fps_approx_bucketed: mem_to_leaf and points must be on the same device");
    TORCH_CHECK(node_aabbs.device() == points.device(), "fps_approx_bucketed: node_aabbs and points must be on the same device");

    if (points.size(2) == 2) {
        return fps_approx_bucketed_impl<2>(
            points, seed_indices, sorted_indices, left_child_mem, right_child_mem,
            mem_to_leaf, node_aabbs, num_real_nodes, leaf_level, M, bucket_size,
            refresh_interval, candidates_per_round, anchors_per_round,
            static_cast<float>(alpha), use_graph
        );
    }
    return fps_approx_bucketed_impl<3>(
        points, seed_indices, sorted_indices, left_child_mem, right_child_mem,
        mem_to_leaf, node_aabbs, num_real_nodes, leaf_level, M, bucket_size,
        refresh_interval, candidates_per_round, anchors_per_round,
        static_cast<float>(alpha), use_graph
    );
}

// ---------------------------------------------------------------------------
// C++/pybind entry points for maintained FPS routes
// ---------------------------------------------------------------------------

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_exact_bucketed_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    bool use_graph,
    bool enable_pruning
) {
    TORCH_CHECK(points.is_cuda(), "fps_exact_bucketed: points must be a CUDA tensor");
    TORCH_CHECK(seed_indices.is_cuda(), "fps_exact_bucketed: seed_indices must be a CUDA tensor");
    TORCH_CHECK(sorted_indices.is_cuda(), "fps_exact_bucketed: sorted_indices must be a CUDA tensor");
    TORCH_CHECK(left_child_mem.is_cuda(), "fps_exact_bucketed: left_child_mem must be a CUDA tensor");
    TORCH_CHECK(right_child_mem.is_cuda(), "fps_exact_bucketed: right_child_mem must be a CUDA tensor");
    TORCH_CHECK(mem_to_leaf.is_cuda(), "fps_exact_bucketed: mem_to_leaf must be a CUDA tensor");
    TORCH_CHECK(node_aabbs.is_cuda(), "fps_exact_bucketed: node_aabbs must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "fps_exact_bucketed: points must be contiguous");
    TORCH_CHECK(seed_indices.is_contiguous(), "fps_exact_bucketed: seed_indices must be contiguous");
    TORCH_CHECK(sorted_indices.is_contiguous(), "fps_exact_bucketed: sorted_indices must be contiguous");
    TORCH_CHECK(left_child_mem.is_contiguous(), "fps_exact_bucketed: left_child_mem must be contiguous");
    TORCH_CHECK(right_child_mem.is_contiguous(), "fps_exact_bucketed: right_child_mem must be contiguous");
    TORCH_CHECK(mem_to_leaf.is_contiguous(), "fps_exact_bucketed: mem_to_leaf must be contiguous");
    TORCH_CHECK(node_aabbs.is_contiguous(), "fps_exact_bucketed: node_aabbs must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "fps_exact_bucketed: points must be float32");
    TORCH_CHECK(seed_indices.scalar_type() == torch::kInt64, "fps_exact_bucketed: seed_indices must be int64");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "fps_exact_bucketed: sorted_indices must be int64");
    TORCH_CHECK(left_child_mem.scalar_type() == torch::kInt32, "fps_exact_bucketed: left_child_mem must be int32");
    TORCH_CHECK(right_child_mem.scalar_type() == torch::kInt32, "fps_exact_bucketed: right_child_mem must be int32");
    TORCH_CHECK(mem_to_leaf.scalar_type() == torch::kInt32, "fps_exact_bucketed: mem_to_leaf must be int32");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "fps_exact_bucketed: node_aabbs must be float32");
    TORCH_CHECK(points.dim() == 3, "fps_exact_bucketed: points must have shape (B, N, D)");
    TORCH_CHECK(seed_indices.dim() == 1, "fps_exact_bucketed: seed_indices must have shape (B,)");
    TORCH_CHECK(sorted_indices.dim() == 2, "fps_exact_bucketed: sorted_indices must have shape (B, N)");
    TORCH_CHECK(node_aabbs.dim() == 3, "fps_exact_bucketed: node_aabbs must have shape (B, num_real_nodes, 2 * D)");
    TORCH_CHECK(left_child_mem.dim() == 1 && right_child_mem.dim() == 1 && mem_to_leaf.dim() == 1, "fps_exact_bucketed: child/leaf arrays must be 1D");
    TORCH_CHECK(points.size(0) >= 1, "fps_exact_bucketed: batch size must be at least 1");
    TORCH_CHECK(points.size(1) >= 1, "fps_exact_bucketed: points must contain at least one point per sample");
    TORCH_CHECK(points.size(2) == 2 || points.size(2) == 3, "fps_exact_bucketed: D must be 2 or 3");

    const int B = static_cast<int>(points.size(0));
    const int N = static_cast<int>(points.size(1));
    const int D = static_cast<int>(points.size(2));
    TORCH_CHECK(seed_indices.size(0) == B, "fps_exact_bucketed: seed_indices length must match batch size");
    TORCH_CHECK(sorted_indices.size(0) == B && sorted_indices.size(1) == N, "fps_exact_bucketed: sorted_indices shape must match points");
    TORCH_CHECK(node_aabbs.size(0) == B && node_aabbs.size(1) == num_real_nodes && node_aabbs.size(2) == 2 * D, "fps_exact_bucketed: node_aabbs shape mismatch");
    TORCH_CHECK(left_child_mem.size(0) == num_real_nodes, "fps_exact_bucketed: left_child_mem length must match num_real_nodes");
    TORCH_CHECK(right_child_mem.size(0) == num_real_nodes, "fps_exact_bucketed: right_child_mem length must match num_real_nodes");
    TORCH_CHECK(mem_to_leaf.size(0) == num_real_nodes, "fps_exact_bucketed: mem_to_leaf length must match num_real_nodes");
    TORCH_CHECK(leaf_level == implicit_bvh::tree::leaf_level(N), "fps_exact_bucketed: leaf_level must match N");
    TORCH_CHECK(M >= 1 && M <= N, "fps_exact_bucketed: target token count must be in [1, N]");
    TORCH_CHECK(bucket_size >= 1, "fps_exact_bucketed: bucket_size must be >= 1");

    c10::cuda::CUDAGuard device_guard(points.device());
    TORCH_CHECK(seed_indices.device() == points.device(), "fps_exact_bucketed: seed_indices and points must be on the same device");
    TORCH_CHECK(sorted_indices.device() == points.device(), "fps_exact_bucketed: sorted_indices and points must be on the same device");
    TORCH_CHECK(left_child_mem.device() == points.device(), "fps_exact_bucketed: left_child_mem and points must be on the same device");
    TORCH_CHECK(right_child_mem.device() == points.device(), "fps_exact_bucketed: right_child_mem and points must be on the same device");
    TORCH_CHECK(mem_to_leaf.device() == points.device(), "fps_exact_bucketed: mem_to_leaf and points must be on the same device");
    TORCH_CHECK(node_aabbs.device() == points.device(), "fps_exact_bucketed: node_aabbs and points must be on the same device");

    if (points.size(2) == 2) {
        return fps_exact_bucketed_impl<2>(
            points, seed_indices, sorted_indices, left_child_mem, right_child_mem,
            mem_to_leaf, node_aabbs, num_real_nodes, leaf_level, M, bucket_size,
            use_graph, enable_pruning
        );
    }
    return fps_exact_bucketed_impl<3>(
        points, seed_indices, sorted_indices, left_child_mem, right_child_mem,
        mem_to_leaf, node_aabbs, num_real_nodes, leaf_level, M, bucket_size,
        use_graph, enable_pruning
    );
}





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_exact_full_scan_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    int M
) {
    TORCH_CHECK(points.is_cuda(), "fps_exact_full_scan: points must be CUDA");
    TORCH_CHECK(seed_indices.is_cuda(), "fps_exact_full_scan: seed_indices must be CUDA");
    TORCH_CHECK(points.is_contiguous() && seed_indices.is_contiguous(), "fps_exact_full_scan: tensors must be contiguous");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "fps_exact_full_scan: points must be float32");
    TORCH_CHECK(seed_indices.scalar_type() == torch::kInt64, "fps_exact_full_scan: seed_indices must be int64");
    TORCH_CHECK(points.dim() == 3 && seed_indices.dim() == 1, "fps_exact_full_scan: shape mismatch");
    TORCH_CHECK(points.size(2) == 2 || points.size(2) == 3, "fps_exact_full_scan: D must be 2 or 3");
    TORCH_CHECK(seed_indices.size(0) == points.size(0), "fps_exact_full_scan: seed_indices length mismatch");
    TORCH_CHECK(M >= 1 && M <= points.size(1), "fps_exact_full_scan: M must be in [1, N]");
    c10::cuda::CUDAGuard device_guard(points.device());
    if (points.size(2) == 2) {
        return fps_exact_full_scan_impl<2>(points, seed_indices, M);
    }
    return fps_exact_full_scan_impl<3>(points, seed_indices, M);
}
