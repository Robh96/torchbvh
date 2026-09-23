// Narrow order-only Morton sorting for batched and routed queries.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cstdint>
#include <tuple>

#include "morton.cuh"
#include <cub/device/device_segmented_radix_sort.cuh>


// One thread per (b, m): write Morton code for queries[b, m, :] to codes[b*M + m].
// Codes are stored as uint64_t (values fit in 30 or 32 bits; int64 storage in PyTorch).
template <int D>
__global__ void query_morton_u32_kernel(
    const float* __restrict__ queries,
    uint32_t* __restrict__ codes,
    int B, int M,
    const float* __restrict__ scene_min,
    const float* __restrict__ scene_max
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(B) * M) return;
    const int b = static_cast<int>(idx / M);
    const float* q = queries + idx * D;
    const float* lo = scene_min + b * D;
    const float* hi = scene_max + b * D;
    if constexpr (D == 2) {
        codes[idx] = implicit_bvh::morton::morton_encode_2d(
            q[0], q[1], make_float2(lo[0], lo[1]), make_float2(hi[0], hi[1]));
    } else {
        codes[idx] = implicit_bvh::morton::morton_encode_3d(
            q[0], q[1], q[2],
            make_float3(lo[0], lo[1], lo[2]), make_float3(hi[0], hi[1], hi[2]));
    }
}

__global__ void fill_iota_i32_kernel(int* out, int M, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = idx % M;
}

__global__ void fill_offsets_i32_kernel(int* offsets, int B, int M) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= B) offsets[idx] = idx * M;
}

__global__ void widen_i32_kernel(const int* input, int64_t* output, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) output[idx] = static_cast<int64_t>(input[idx]);
}


// One thread per (b, m): encode against the bounds for the selected route and
// store that route above the Morton bits. Sorting the composite key therefore
// groups routes without losing spatial locality inside either group.
template <int D>
__global__ void routed_query_morton_kernel(
    const float* __restrict__ queries,
    const bool* __restrict__ routes,
    uint64_t* __restrict__ codes,
    int B, int M,
    const float* __restrict__ true_scene_min,
    const float* __restrict__ true_scene_max,
    const float* __restrict__ false_scene_min,
    const float* __restrict__ false_scene_max
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * M) return;
    const int b = (int)(idx / M);
    const int m = (int)(idx - (int64_t)b * M);
    const bool route = routes[idx];
    const float* q = queries + ((int64_t)b * M + m) * D;
    const float* lo = (route ? true_scene_min : false_scene_min) + b * D;
    const float* hi = (route ? true_scene_max : false_scene_max) + b * D;

    uint64_t spatial_code;
    if constexpr (D == 2) {
        spatial_code = (uint64_t)implicit_bvh::morton::morton_encode_2d(
            q[0], q[1],
            make_float2(lo[0], lo[1]),
            make_float2(hi[0], hi[1]));
    } else {
        spatial_code = (uint64_t)implicit_bvh::morton::morton_encode_3d(
            q[0], q[1], q[2],
            make_float3(lo[0], lo[1], lo[2]),
            make_float3(hi[0], hi[1], hi[2]));
    }
    constexpr int spatial_bits = (D == 3) ? 30 : 32;
    codes[idx] = spatial_code | ((uint64_t)route << spatial_bits);
}


// One thread per flat index: fill out[idx] = idx % M (iota within each segment).
__global__ void fill_iota_segmented_kernel(
    int64_t* __restrict__ out, int64_t M, int64_t total
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = idx % M;
}


// Return only each segment's sorted original indices; no inverse map is needed.
torch::Tensor segmented_sort_codes(
    torch::Tensor codes,
    int B,
    int M,
    int end_bit,
    const torch::TensorOptions& options
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = (int64_t)B * M;
    auto i64_opts = options.dtype(torch::kInt64);
    auto byte_opts = options.dtype(torch::kByte);
    auto codes_out = torch::empty({total}, i64_opts);
    auto values_in = torch::empty({total}, i64_opts);
    auto sort_perm = torch::empty({total}, i64_opts);

    constexpr int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    fill_iota_segmented_kernel<<<blocks, threads, 0, stream>>>(
        values_in.data_ptr<int64_t>(), (int64_t)M, total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto offsets = torch::arange(B + 1, options.dtype(torch::kInt32)) * M;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes,
        reinterpret_cast<const uint64_t*>(codes.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(codes_out.data_ptr<int64_t>()),
        values_in.data_ptr<int64_t>(), sort_perm.data_ptr<int64_t>(),
        (int)total, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream
    );
    auto temp_storage = torch::empty({(int64_t)(temp_bytes + 1)}, byte_opts);
    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data_ptr(), temp_bytes,
        reinterpret_cast<const uint64_t*>(codes.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(codes_out.data_ptr<int64_t>()),
        values_in.data_ptr<int64_t>(), sort_perm.data_ptr<int64_t>(),
        (int)total, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sort_perm;
}


template <int D>
torch::Tensor morton_sort_routed_queries_batched_impl(
    torch::Tensor queries,
    torch::Tensor routes,
    torch::Tensor true_scene_min,
    torch::Tensor true_scene_max,
    torch::Tensor false_scene_min,
    torch::Tensor false_scene_max
) {
    c10::cuda::CUDAGuard device_guard(queries.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int B = (int)queries.size(0);
    const int M = (int)queries.size(1);
    const int64_t total = (int64_t)B * M;
    auto codes = torch::empty({total}, queries.options().dtype(torch::kInt64));

    constexpr int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    routed_query_morton_kernel<D><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(), routes.data_ptr<bool>(),
        reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>()), B, M,
        true_scene_min.data_ptr<float>(), true_scene_max.data_ptr<float>(),
        false_scene_min.data_ptr<float>(), false_scene_max.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int end_bit = ((D == 3) ? 30 : 32) + 1;
    return segmented_sort_codes(codes, B, M, end_bit, queries.options()).view({B, M});
}

template <int D>
torch::Tensor morton_sort_queries_narrow_impl(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    c10::cuda::CUDAGuard device_guard(queries.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int B = static_cast<int>(queries.size(0));
    const int M = static_cast<int>(queries.size(1));
    const int total = B * M;
    auto i32_options = queries.options().dtype(torch::kInt32);
    auto i64_options = queries.options().dtype(torch::kInt64);
    auto byte_options = queries.options().dtype(torch::kByte);
    auto codes_in = torch::empty({total}, i32_options);
    auto codes_out = torch::empty({total}, i32_options);
    auto values_in = torch::empty({total}, i32_options);
    auto values_out = torch::empty({total}, i32_options);
    auto offsets = torch::empty({B + 1}, i32_options);
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    query_morton_u32_kernel<D><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(), reinterpret_cast<uint32_t*>(codes_in.data_ptr<int>()),
        B, M, scene_min.data_ptr<float>(), scene_max.data_ptr<float>());
    fill_iota_i32_kernel<<<blocks, threads, 0, stream>>>(values_in.data_ptr<int>(), M, total);
    fill_offsets_i32_kernel<<<(B + 1 + threads - 1) / threads, threads, 0, stream>>>(
        offsets.data_ptr<int>(), B, M);
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
    auto order = torch::empty({total}, i64_options);
    widen_i32_kernel<<<blocks, threads, 0, stream>>>(
        values_out.data_ptr<int>(), order.data_ptr<int64_t>(), total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return order.view({B, M});
}


torch::Tensor morton_sort_queries_narrow_cuda(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    const char* op = "morton_sort_queries_narrow";
    TORCH_CHECK(queries.is_cuda() && queries.is_contiguous()
                && queries.scalar_type() == torch::kFloat32,
                op, ": queries must be contiguous CUDA float32");
    TORCH_CHECK(scene_min.is_cuda() && scene_max.is_cuda()
                && scene_min.is_contiguous() && scene_max.is_contiguous(),
                op, ": bounds must be contiguous CUDA tensors");
    TORCH_CHECK(queries.dim() == 3 && (queries.size(2) == 2 || queries.size(2) == 3),
                op, ": queries must have shape (B, M, 2|3)");
    TORCH_CHECK(scene_min.sizes() == scene_max.sizes()
                && scene_min.size(0) == queries.size(0)
                && scene_min.size(1) == queries.size(2),
                op, ": bounds shape mismatch");
    if (queries.size(2) == 2) {
        return morton_sort_queries_narrow_impl<2>(queries, scene_min, scene_max);
    }
    return morton_sort_queries_narrow_impl<3>(queries, scene_min, scene_max);
}


torch::Tensor morton_sort_routed_queries_batched_cuda(
    torch::Tensor queries,
    torch::Tensor routes,
    torch::Tensor true_scene_min,
    torch::Tensor true_scene_max,
    torch::Tensor false_scene_min,
    torch::Tensor false_scene_max
) {
    const char* op = "morton_sort_routed_queries_batched";
    TORCH_CHECK(queries.is_cuda() && routes.is_cuda(), op, ": queries and routes must be CUDA tensors");
    TORCH_CHECK(true_scene_min.is_cuda() && true_scene_max.is_cuda() &&
                false_scene_min.is_cuda() && false_scene_max.is_cuda(),
                op, ": scene bounds must be CUDA tensors");
    TORCH_CHECK(queries.is_contiguous() && routes.is_contiguous(), op, ": queries and routes must be contiguous");
    TORCH_CHECK(true_scene_min.is_contiguous() && true_scene_max.is_contiguous() &&
                false_scene_min.is_contiguous() && false_scene_max.is_contiguous(),
                op, ": scene bounds must be contiguous");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, op, ": queries must be float32");
    TORCH_CHECK(routes.scalar_type() == torch::kBool, op, ": routes must be bool");
    for (const auto& bound : {true_scene_min, true_scene_max, false_scene_min, false_scene_max}) {
        TORCH_CHECK(bound.scalar_type() == torch::kFloat32, op, ": scene bounds must be float32");
        TORCH_CHECK(bound.device() == queries.device(), op, ": all inputs must share a device");
    }
    TORCH_CHECK(routes.device() == queries.device(), op, ": all inputs must share a device");
    TORCH_CHECK(queries.dim() == 3, op, ": queries must have shape (B, M, D)");
    TORCH_CHECK(routes.dim() == 2 && routes.size(0) == queries.size(0) && routes.size(1) == queries.size(1),
                op, ": routes must have shape (B, M)");
    const int D = (int)queries.size(2);
    TORCH_CHECK(D == 2 || D == 3, op, ": D must be 2 or 3");
    for (const auto& bound : {true_scene_min, true_scene_max, false_scene_min, false_scene_max}) {
        TORCH_CHECK(bound.dim() == 2 && bound.size(0) == queries.size(0) && bound.size(1) == D,
                    op, ": each scene bound must have shape (B, D)");
    }
    if (D == 2) {
        return morton_sort_routed_queries_batched_impl<2>(
            queries, routes, true_scene_min, true_scene_max, false_scene_min, false_scene_max);
    }
    return morton_sort_routed_queries_batched_impl<3>(
        queries, routes, true_scene_min, true_scene_max, false_scene_min, false_scene_max);
}
