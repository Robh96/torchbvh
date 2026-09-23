#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cfloat>
#include <tuple>


namespace {

constexpr float kIntersectionEpsilon = 8.0f * FLT_EPSILON;
constexpr int kTraversalStackCapacity = 64;
constexpr int kCachedTraversalStackCapacity = 32;

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

template <int D, int V>
__global__ void raytrace_batched_cached_kernel(
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
    float* __restrict__ out_points,
    bool* __restrict__ out_mask,
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

    const int64_t output_offset = static_cast<int64_t>(batch) * query_count + query;
    const int64_t ray_offset = output_offset * D;
    const float* origin = origins + ray_offset;
    const float* direction = directions + ray_offset;
    const float* sample_aabbs = node_aabbs + static_cast<int64_t>(batch) * num_real_nodes * 2 * D;
    const int64_t* sample_indices = sorted_indices + static_cast<int64_t>(batch) * primitive_count;
    const float* sample_primitives = primitives + static_cast<int64_t>(batch) * primitive_count * V * D;

    float best_t = t_max;
    int64_t best_idx = -1;
    int stack_nodes[kCachedTraversalStackCapacity];
    float stack_entries[kCachedTraversalStackCapacity];
    int stack_size = 0;

    int current = 0;
    float current_entry;
    if (!ray_aabb_entry(origin, direction, sample_aabbs, D, t_min, best_t, &current_entry)) {
        current = -1;
    }

    while (current >= 0) {
        bool descend = false;
        if (current_entry <= best_t) {
            const int leaf_position = mem_to_leaf[current];
            if (leaf_position >= 0) {
                const int64_t original_idx = sample_indices[leaf_position];
                const float* primitive = sample_primitives + original_idx * V * D;
                float candidate_t;
                bool hit;
                if constexpr (D == 2) {
                    hit = intersect_segment_2d(
                        primitive, origin, direction, t_min, best_t, &candidate_t
                    );
                } else {
                    hit = intersect_triangle_3d(
                        primitive, origin, direction, t_min, best_t, &candidate_t
                    );
                }
                if (hit && better_hit(candidate_t, original_idx, best_t, best_idx)) {
                    best_t = candidate_t;
                    best_idx = original_idx;
                }
            } else {
                const int left = left_child_mem[current];
                const int right = right_child_mem[current];
                float left_entry = device_infinity();
                float right_entry = device_infinity();
                const bool left_hit = left >= 0 && ray_aabb_entry(
                    origin, direction, sample_aabbs + left * 2 * D,
                    D, t_min, best_t, &left_entry
                );
                const bool right_hit = right >= 0 && ray_aabb_entry(
                    origin, direction, sample_aabbs + right * 2 * D,
                    D, t_min, best_t, &right_entry
                );

                if (left_hit || right_hit) {
                    int near_node;
                    float near_entry;
                    if (!right_hit || (left_hit && left_entry <= right_entry)) {
                        near_node = left;
                        near_entry = left_entry;
                        if (right_hit) {
                            stack_nodes[stack_size] = right;
                            stack_entries[stack_size++] = right_entry;
                        }
                    } else {
                        near_node = right;
                        near_entry = right_entry;
                        if (left_hit) {
                            stack_nodes[stack_size] = left;
                            stack_entries[stack_size++] = left_entry;
                        }
                    }
                    current = near_node;
                    current_entry = near_entry;
                    descend = true;
                }
            }
        }

        if (descend) {
            continue;
        }
        current = -1;
        while (stack_size > 0) {
            --stack_size;
            if (stack_entries[stack_size] <= best_t) {
                current = stack_nodes[stack_size];
                current_entry = stack_entries[stack_size];
                break;
            }
        }
    }

    const bool hit = best_idx >= 0;
    out_indices[output_offset] = best_idx;
    out_t[output_offset] = hit ? best_t : device_infinity();
    out_mask[output_offset] = hit;
    #pragma unroll
    for (int axis = 0; axis < D; ++axis) {
        out_points[ray_offset + axis] = hit
            ? origin[axis] + best_t * direction[axis]
            : __int_as_float(0x7fffffff);
    }
}

__global__ void segment_raytrace_backward_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ hit_t,
    const float* __restrict__ primitives,
    const float* __restrict__ origins,
    const float* __restrict__ directions,
    const float* __restrict__ grad_t,
    const float* __restrict__ grad_points,
    float* __restrict__ grad_primitives,
    float* __restrict__ grad_origins,
    float* __restrict__ grad_directions,
    int query_count,
    int primitive_count,
    bool need_primitives,
    bool need_origins,
    bool need_directions
) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (query >= query_count) {
        return;
    }
    const int64_t ray_index = static_cast<int64_t>(batch) * query_count + query;
    const int64_t ray_offset = ray_index * 2;
    const int64_t primitive_idx = indices[ray_index];
    if (primitive_idx < 0) {
        if (need_origins) {
            grad_origins[ray_offset] = 0.0f;
            grad_origins[ray_offset + 1] = 0.0f;
        }
        if (need_directions) {
            grad_directions[ray_offset] = 0.0f;
            grad_directions[ray_offset + 1] = 0.0f;
        }
        return;
    }

    const int64_t primitive_offset =
        (static_cast<int64_t>(batch) * primitive_count + primitive_idx) * 4;
    const float* primitive = primitives + primitive_offset;
    const float* origin = origins + ray_offset;
    const float* direction = directions + ray_offset;
    const float ex = primitive[2] - primitive[0];
    const float ey = primitive[3] - primitive[1];
    const float ax = primitive[0] - origin[0];
    const float ay = primitive[1] - origin[1];
    const float denominator = cross2(direction[0], direction[1], ex, ey);
    const float t = hit_t[ray_index];

    const float point_grad_x = grad_points == nullptr ? 0.0f : grad_points[ray_offset];
    const float point_grad_y = grad_points == nullptr ? 0.0f : grad_points[ray_offset + 1];
    float total_t_grad = grad_t == nullptr ? 0.0f : grad_t[ray_index];
    total_t_grad += point_grad_x * direction[0] + point_grad_y * direction[1];

    const float numerator_grad = total_t_grad / denominator;
    const float denominator_grad = -total_t_grad * t / denominator;
    const float grad_ax = numerator_grad * ey;
    const float grad_ay = -numerator_grad * ex;
    const float grad_ex = -numerator_grad * ay - denominator_grad * direction[1];
    const float grad_ey = numerator_grad * ax + denominator_grad * direction[0];

    if (need_primitives) {
        atomicAdd(grad_primitives + primitive_offset, grad_ax - grad_ex);
        atomicAdd(grad_primitives + primitive_offset + 1, grad_ay - grad_ey);
        atomicAdd(grad_primitives + primitive_offset + 2, grad_ex);
        atomicAdd(grad_primitives + primitive_offset + 3, grad_ey);
    }
    if (need_origins) {
        grad_origins[ray_offset] = point_grad_x - grad_ax;
        grad_origins[ray_offset + 1] = point_grad_y - grad_ay;
    }
    if (need_directions) {
        grad_directions[ray_offset] =
            t * point_grad_x + denominator_grad * ey;
        grad_directions[ray_offset + 1] =
            t * point_grad_y - denominator_grad * ex;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_raytrace_batched_cached(
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
    auto points = torch::empty({B, Q, 2}, origins.options());
    auto mask = torch::empty({B, Q}, origins.options().dtype(torch::kBool));
    if (Q == 0) {
        return std::make_tuple(indices, hit_t, points, mask);
    }
    constexpr int threads = 128;
    const dim3 blocks((Q + threads - 1) / threads, B);
    raytrace_batched_cached_kernel<2, 2><<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        node_aabbs.data_ptr<float>(), sorted_indices.data_ptr<int64_t>(),
        left_child_mem.data_ptr<int>(), right_child_mem.data_ptr<int>(),
        mem_to_leaf.data_ptr<int>(), primitives.data_ptr<float>(),
        origins.data_ptr<float>(), directions.data_ptr<float>(),
        indices.data_ptr<int64_t>(), hit_t.data_ptr<float>(), points.data_ptr<float>(),
        mask.data_ptr<bool>(), Q, F, num_real_nodes, t_min, t_max
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(indices, hit_t, points, mask);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
raytrace_batched_cached_cuda(
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
    TORCH_CHECK(node_aabbs.is_cuda() && sorted_indices.is_cuda(), "raytrace_batched_cached: BVH tensors must be CUDA tensors");
    TORCH_CHECK(left_child_mem.is_cuda() && right_child_mem.is_cuda() && mem_to_leaf.is_cuda(), "raytrace_batched_cached: traversal tensors must be CUDA tensors");
    TORCH_CHECK(primitives.is_cuda() && origins.is_cuda() && directions.is_cuda(), "raytrace_batched_cached: ray tensors must be CUDA tensors");
    TORCH_CHECK(node_aabbs.is_contiguous() && sorted_indices.is_contiguous(), "raytrace_batched_cached: BVH tensors must be contiguous");
    TORCH_CHECK(left_child_mem.is_contiguous() && right_child_mem.is_contiguous() && mem_to_leaf.is_contiguous(), "raytrace_batched_cached: traversal tensors must be contiguous");
    TORCH_CHECK(primitives.is_contiguous() && origins.is_contiguous() && directions.is_contiguous(), "raytrace_batched_cached: geometry and rays must be contiguous");
    TORCH_CHECK(node_aabbs.scalar_type() == torch::kFloat32, "raytrace_batched_cached: node_aabbs must be float32");
    TORCH_CHECK(primitives.scalar_type() == torch::kFloat32 && origins.scalar_type() == torch::kFloat32 && directions.scalar_type() == torch::kFloat32, "raytrace_batched_cached: geometry and rays must be float32");
    TORCH_CHECK(sorted_indices.scalar_type() == torch::kInt64, "raytrace_batched_cached: sorted_indices must be int64");
    TORCH_CHECK(left_child_mem.scalar_type() == torch::kInt32 && right_child_mem.scalar_type() == torch::kInt32 && mem_to_leaf.scalar_type() == torch::kInt32, "raytrace_batched_cached: traversal tensors must be int32");
    TORCH_CHECK(primitives.dim() == 4 && primitives.size(2) == 2 && primitives.size(3) == 2, "raytrace_batched_cached: expected batched 2-D segments");
    TORCH_CHECK(origins.dim() == 3 && origins.size(2) == 2 && origins.sizes() == directions.sizes(), "raytrace_batched_cached: rays must have shape (B, Q, 2)");
    TORCH_CHECK(node_aabbs.dim() == 3 && node_aabbs.size(2) == 4 && sorted_indices.dim() == 2, "raytrace_batched_cached: invalid BVH shapes");
    TORCH_CHECK(primitives.size(0) == origins.size(0) && node_aabbs.size(0) == origins.size(0) && sorted_indices.size(0) == origins.size(0), "raytrace_batched_cached: batch sizes must match");
    TORCH_CHECK(node_aabbs.size(1) == num_real_nodes, "raytrace_batched_cached: num_real_nodes does not match BVH");
    TORCH_CHECK(left_child_mem.numel() == num_real_nodes && right_child_mem.numel() == num_real_nodes && mem_to_leaf.numel() == num_real_nodes, "raytrace_batched_cached: traversal arrays do not match BVH");
    TORCH_CHECK(sorted_indices.size(1) == primitives.size(1), "raytrace_batched_cached: primitive count does not match BVH");
    TORCH_CHECK(t_min >= 0.0 && t_max > t_min, "raytrace_batched_cached: expected 0 <= t_min < t_max");

    c10::cuda::CUDAGuard device_guard(origins.device());
    TORCH_CHECK(node_aabbs.device() == origins.device() && sorted_indices.device() == origins.device(), "raytrace_batched_cached: BVH and rays must share a device");
    TORCH_CHECK(left_child_mem.device() == origins.device() && right_child_mem.device() == origins.device() && mem_to_leaf.device() == origins.device(), "raytrace_batched_cached: traversal data and rays must share a device");
    TORCH_CHECK(primitives.device() == origins.device() && directions.device() == origins.device(), "raytrace_batched_cached: geometry and rays must share a device");
    return launch_raytrace_batched_cached(
        node_aabbs, sorted_indices, left_child_mem, right_child_mem, mem_to_leaf,
        primitives, origins, directions, num_real_nodes,
        static_cast<float>(t_min), static_cast<float>(t_max)
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
raytrace_segment_backward_cuda(
    torch::Tensor indices,
    torch::Tensor hit_t,
    torch::Tensor primitives,
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor grad_t,
    torch::Tensor grad_points,
    bool need_primitives,
    bool need_origins,
    bool need_directions
) {
    TORCH_CHECK(indices.is_cuda() && hit_t.is_cuda() && primitives.is_cuda(), "raytrace_segment_backward: tensors must be CUDA tensors");
    TORCH_CHECK(origins.is_cuda() && directions.is_cuda(), "raytrace_segment_backward: tensors must be CUDA tensors");
    TORCH_CHECK(indices.is_contiguous() && hit_t.is_contiguous() && primitives.is_contiguous(), "raytrace_segment_backward: tensors must be contiguous");
    TORCH_CHECK(origins.is_contiguous() && directions.is_contiguous(), "raytrace_segment_backward: tensors must be contiguous");
    TORCH_CHECK(grad_t.numel() == 0 || grad_t.is_contiguous(), "raytrace_segment_backward: grad_t must be contiguous");
    TORCH_CHECK(grad_points.numel() == 0 || grad_points.is_contiguous(), "raytrace_segment_backward: grad_points must be contiguous");

    c10::cuda::CUDAGuard device_guard(origins.device());
    auto grad_primitives = need_primitives
        ? torch::zeros_like(primitives)
        : torch::empty({0}, primitives.options());
    auto grad_origins = need_origins
        ? torch::empty_like(origins)
        : torch::empty({0}, origins.options());
    auto grad_directions = need_directions
        ? torch::empty_like(directions)
        : torch::empty({0}, directions.options());
    const int B = static_cast<int>(origins.size(0));
    const int Q = static_cast<int>(origins.size(1));
    if (Q > 0 && (need_primitives || need_origins || need_directions)) {
        constexpr int threads = 128;
        const dim3 blocks((Q + threads - 1) / threads, B);
        segment_raytrace_backward_kernel<<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            indices.data_ptr<int64_t>(), hit_t.data_ptr<float>(),
            primitives.data_ptr<float>(), origins.data_ptr<float>(),
            directions.data_ptr<float>(),
            grad_t.numel() == 0 ? nullptr : grad_t.data_ptr<float>(),
            grad_points.numel() == 0 ? nullptr : grad_points.data_ptr<float>(),
            need_primitives ? grad_primitives.data_ptr<float>() : nullptr,
            need_origins ? grad_origins.data_ptr<float>() : nullptr,
            need_directions ? grad_directions.data_ptr<float>() : nullptr,
            Q, static_cast<int>(primitives.size(1)),
            need_primitives, need_origins, need_directions
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(grad_primitives, grad_origins, grad_directions);
}
