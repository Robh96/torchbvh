#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cfloat>
#include <tuple>


namespace {

constexpr float kIntersectionEpsilon = 8.0f * FLT_EPSILON;
constexpr int kTraversalStackCapacity = 64;

__device__ inline float device_infinity() {
    return __int_as_float(0x7f800000);
}

__device__ inline float cross2(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

__device__ inline bool ray_aabb_entry(
    const float* __restrict__ origin,
    const float* __restrict__ direction,
    const float* __restrict__ aabb,
    int dim,
    float t_min,
    float t_max,
    float* __restrict__ entry
) {
    float lo = t_min;
    float hi = t_max;
    for (int axis = 0; axis < dim; ++axis) {
        const float d = direction[axis];
        const float o = origin[axis];
        const float slab_lo = aabb[axis];
        const float slab_hi = aabb[dim + axis];
        if (d == 0.0f) {
            if (o < slab_lo || o > slab_hi) {
                return false;
            }
            continue;
        }

        float t0 = (slab_lo - o) / d;
        float t1 = (slab_hi - o) / d;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        lo = fmaxf(lo, t0);
        hi = fminf(hi, t1);
        if (hi < lo) {
            return false;
        }
    }
    *entry = lo;
    return true;
}

__device__ inline bool intersect_segment_2d(
    const float* __restrict__ primitive,
    const float* __restrict__ origin,
    const float* __restrict__ direction,
    float t_min,
    float t_max,
    float* __restrict__ hit_t
) {
    const float ex = primitive[2] - primitive[0];
    const float ey = primitive[3] - primitive[1];
    const float dir_sq = direction[0] * direction[0] + direction[1] * direction[1];
    const float edge_sq = ex * ex + ey * ey;
    if (dir_sq == 0.0f || edge_sq == 0.0f) {
        return false;
    }

    const float denominator = cross2(direction[0], direction[1], ex, ey);
    const float denominator_tolerance = kIntersectionEpsilon * sqrtf(dir_sq * edge_sq);
    if (fabsf(denominator) <= denominator_tolerance) {
        return false;
    }

    const float ax = primitive[0] - origin[0];
    const float ay = primitive[1] - origin[1];
    const float t = cross2(ax, ay, ex, ey) / denominator;
    const float u = cross2(ax, ay, direction[0], direction[1]) / denominator;
    if (t < t_min || t > t_max || u < -kIntersectionEpsilon || u > 1.0f + kIntersectionEpsilon) {
        return false;
    }
    *hit_t = t;
    return true;
}

__device__ inline void cross3(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline float dot3(
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline bool intersect_triangle_3d(
    const float* __restrict__ primitive,
    const float* __restrict__ origin,
    const float* __restrict__ direction,
    float t_min,
    float t_max,
    float* __restrict__ hit_t
) {
    float edge1[3];
    float edge2[3];
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        edge1[axis] = primitive[3 + axis] - primitive[axis];
        edge2[axis] = primitive[6 + axis] - primitive[axis];
    }

    float normal[3];
    cross3(edge1, edge2, normal);
    const float direction_sq = dot3(direction, direction);
    const float normal_sq = dot3(normal, normal);
    if (direction_sq == 0.0f || normal_sq == 0.0f) {
        return false;
    }

    float pvec[3];
    cross3(direction, edge2, pvec);
    const float determinant = dot3(edge1, pvec);
    const float determinant_tolerance = kIntersectionEpsilon * sqrtf(direction_sq * normal_sq);
    if (fabsf(determinant) <= determinant_tolerance) {
        return false;
    }

    const float inverse_determinant = 1.0f / determinant;
    float tvec[3];
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        tvec[axis] = origin[axis] - primitive[axis];
    }
    const float u = dot3(tvec, pvec) * inverse_determinant;
    if (u < -kIntersectionEpsilon || u > 1.0f + kIntersectionEpsilon) {
        return false;
    }

    float qvec[3];
    cross3(tvec, edge1, qvec);
    const float v = dot3(direction, qvec) * inverse_determinant;
    if (v < -kIntersectionEpsilon || u + v > 1.0f + kIntersectionEpsilon) {
        return false;
    }

    const float t = dot3(edge2, qvec) * inverse_determinant;
    if (t < t_min || t > t_max) {
        return false;
    }
    *hit_t = t;
    return true;
}

__device__ inline bool better_hit(float candidate_t, int64_t candidate_idx, float best_t, int64_t best_idx) {
    if (best_idx < 0) {
        return true;
    }
    const float tolerance = kIntersectionEpsilon * fmaxf(1.0f, fmaxf(fabsf(candidate_t), fabsf(best_t)));
    if (candidate_t < best_t - tolerance) {
        return true;
    }
    return fabsf(candidate_t - best_t) <= tolerance && candidate_idx < best_idx;
}

template <int D, int V>
__global__ void raytrace_batched_kernel(
    const float* __restrict__ node_aabbs,
    const int64_t* __restrict__ sorted_indices,
    const int* __restrict__ left_child_mem,
    const int* __restrict__ right_child_mem,
    const int* __restrict__ mem_to_leaf,
    const float* __restrict__ primitives,
    const float* __restrict__ origins,
    const float* __restrict__ directions,
    int64_t* __restrict__ out_indices,
    float* __restrict__ out_t,
    int query_count,
    int primitive_count,
    int num_real_nodes,
    float t_min,
    float t_max
) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (query >= query_count) {
        return;
    }

    const int64_t ray_offset = (static_cast<int64_t>(batch) * query_count + query) * D;
    const float* origin = origins + ray_offset;
    const float* direction = directions + ray_offset;
    const float* sample_aabbs = node_aabbs + static_cast<int64_t>(batch) * num_real_nodes * 2 * D;
    const int64_t* sample_indices = sorted_indices + static_cast<int64_t>(batch) * primitive_count;
    const float* sample_primitives = primitives + static_cast<int64_t>(batch) * primitive_count * V * D;

    float best_t = t_max;
    int64_t best_idx = -1;
    int stack[kTraversalStackCapacity];
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const int mem_idx = stack[--stack_size];
        float node_entry;
        if (!ray_aabb_entry(
                origin, direction, sample_aabbs + mem_idx * 2 * D,
                D, t_min, best_t, &node_entry)) {
            continue;
        }

        const int leaf_position = mem_to_leaf[mem_idx];
        if (leaf_position >= 0) {
            const int64_t original_idx = sample_indices[leaf_position];
            const float* primitive = sample_primitives + original_idx * V * D;
            float candidate_t;
            bool hit;
            if constexpr (D == 2) {
                hit = intersect_segment_2d(primitive, origin, direction, t_min, best_t, &candidate_t);
            } else {
                hit = intersect_triangle_3d(primitive, origin, direction, t_min, best_t, &candidate_t);
            }
            if (hit && better_hit(candidate_t, original_idx, best_t, best_idx)) {
                best_t = candidate_t;
                best_idx = original_idx;
            }
            continue;
        }

        const int left = left_child_mem[mem_idx];
        const int right = right_child_mem[mem_idx];
        float left_entry = device_infinity();
        float right_entry = device_infinity();
        const bool left_hit = left >= 0 && ray_aabb_entry(
            origin, direction, sample_aabbs + left * 2 * D, D, t_min, best_t, &left_entry
        );
        const bool right_hit = right >= 0 && ray_aabb_entry(
            origin, direction, sample_aabbs + right * 2 * D, D, t_min, best_t, &right_entry
        );

        if (left_entry <= right_entry) {
            if (right_hit) stack[stack_size++] = right;
            if (left_hit) stack[stack_size++] = left;
        } else {
            if (left_hit) stack[stack_size++] = left;
            if (right_hit) stack[stack_size++] = right;
        }
    }

    const int64_t output_offset = static_cast<int64_t>(batch) * query_count + query;
    out_indices[output_offset] = best_idx;
    out_t[output_offset] = best_idx >= 0 ? best_t : device_infinity();
}

template <int D, int V>
std::tuple<torch::Tensor, torch::Tensor> launch_raytrace_batched(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor primitives,
    torch::Tensor origins,
    torch::Tensor directions,
    int num_real_nodes,
    float t_min,
    float t_max
) {
    const int B = static_cast<int>(origins.size(0));
    const int Q = static_cast<int>(origins.size(1));
    const int F = static_cast<int>(primitives.size(1));
    auto indices = torch::empty({B, Q}, origins.options().dtype(torch::kInt64));
    auto hit_t = torch::empty({B, Q}, origins.options());
    if (Q == 0) {
        return std::make_tuple(indices, hit_t);
    }

    constexpr int threads = 128;
    const dim3 blocks((Q + threads - 1) / threads, B);
    raytrace_batched_kernel<D, V><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_aabbs.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        left_child_mem.data_ptr<int>(),
        right_child_mem.data_ptr<int>(),
        mem_to_leaf.data_ptr<int>(),
        primitives.data_ptr<float>(),
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        hit_t.data_ptr<float>(),
        Q,
        F,
        num_real_nodes,
        t_min,
        t_max
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, hit_t);
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor> raytrace_batched_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor primitives,
    torch::Tensor origins,
    torch::Tensor directions,
    int num_real_nodes,
    double t_min,
    double t_max
) {
    TORCH_CHECK(node_aabbs.is_cuda() && sorted_indices.is_cuda(), "raytrace_batched: BVH tensors must be CUDA tensors");
    TORCH_CHECK(left_child_mem.is_cuda() && right_child_mem.is_cuda() && mem_to_leaf.is_cuda(), "raytrace_batched: traversal tensors must be CUDA tensors");
    TORCH_CHECK(primitives.is_cuda() && origins.is_cuda() && directions.is_cuda(), "raytrace_batched: ray tensors must be CUDA tensors");
    TORCH_CHECK(node_aabbs.is_contiguous() && sorted_indices.is_contiguous(), "raytrace_batched: BVH tensors must be contiguous");
    TORCH_CHECK(left_child_mem.is_contiguous() && right_child_mem.is_contiguous() && mem_to_leaf.is_contiguous(), "raytrace_batched: traversal tensors must be contiguous");
    TORCH_CHECK(primitives.is_contiguous() && origins.is_contiguous() && directions.is_contiguous(), "raytrace_batched: ray tensors must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "raytrace_batched: node_aabbs must be float32");
    TORCH_CHECK(primitives.scalar_type() == torch::kFloat32 && origins.scalar_type() == torch::kFloat32 && directions.scalar_type() == torch::kFloat32, "raytrace_batched: geometry and rays must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "raytrace_batched: sorted_indices must be int64");
    TORCH_CHECK(left_child_mem.scalar_type() == torch::kInt32 && right_child_mem.scalar_type() == torch::kInt32 && mem_to_leaf.scalar_type() == torch::kInt32, "raytrace_batched: traversal tensors must be int32");
    TORCH_CHECK(primitives.dim() == 4, "raytrace_batched: primitives must have shape (B, F, V, D)");
    TORCH_CHECK(origins.dim() == 3 && directions.dim() == 3, "raytrace_batched: rays must have shape (B, Q, D)");
    TORCH_CHECK(origins.sizes() == directions.sizes(), "raytrace_batched: origins and directions must have identical shapes");
    TORCH_CHECK(node_aabbs.dim() == 3 && sorted_indices.dim() == 2, "raytrace_batched: invalid batched BVH shapes");
    TORCH_CHECK(primitives.size(0) == origins.size(0), "raytrace_batched: primitive and ray batch sizes must match");
    TORCH_CHECK(node_aabbs.size(0) == origins.size(0) && sorted_indices.size(0) == origins.size(0), "raytrace_batched: BVH and ray batch sizes must match");
    TORCH_CHECK(node_aabbs.size(1) == num_real_nodes, "raytrace_batched: num_real_nodes does not match BVH");
    TORCH_CHECK(left_child_mem.numel() == num_real_nodes && right_child_mem.numel() == num_real_nodes && mem_to_leaf.numel() == num_real_nodes, "raytrace_batched: traversal arrays do not match BVH");
    TORCH_CHECK(sorted_indices.size(1) == primitives.size(1), "raytrace_batched: primitive count does not match BVH");

    const int V = static_cast<int>(primitives.size(2));
    const int D = static_cast<int>(primitives.size(3));
    TORCH_CHECK(origins.size(2) == D, "raytrace_batched: ray dimension must match primitives");
    TORCH_CHECK(node_aabbs.size(2) == 2 * D, "raytrace_batched: AABB dimension must match primitives");
    TORCH_CHECK((V == 2 && D == 2) || (V == 3 && D == 3), "raytrace_batched: expected 2-D segments or 3-D triangles");
    TORCH_CHECK(t_min >= 0.0 && t_max > t_min, "raytrace_batched: expected 0 <= t_min < t_max");

    c10::cuda::CUDAGuard device_guard(origins.device());
    TORCH_CHECK(node_aabbs.device() == origins.device() && sorted_indices.device() == origins.device(), "raytrace_batched: BVH and rays must share a device");
    TORCH_CHECK(left_child_mem.device() == origins.device() && right_child_mem.device() == origins.device() && mem_to_leaf.device() == origins.device(), "raytrace_batched: traversal data and rays must share a device");
    TORCH_CHECK(primitives.device() == origins.device() && directions.device() == origins.device(), "raytrace_batched: geometry and rays must share a device");

    if (D == 2) {
        return launch_raytrace_batched<2, 2>(
            node_aabbs, sorted_indices, left_child_mem, right_child_mem, mem_to_leaf,
            primitives, origins, directions, num_real_nodes,
            static_cast<float>(t_min), static_cast<float>(t_max)
        );
    }
    return launch_raytrace_batched<3, 3>(
        node_aabbs, sorted_indices, left_child_mem, right_child_mem, mem_to_leaf,
        primitives, origins, directions, num_real_nodes,
        static_cast<float>(t_min), static_cast<float>(t_max)
    );
}
