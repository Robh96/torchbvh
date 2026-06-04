// Fused Morton encoding + CUB segmented radix sort for batched query reordering.
// Returns (sort_perm, inv_perm) of shape (B, M) dtype int64.
//
// Three kernels on a single stream:
//   1. query_morton_kernel        : (B*M) threads, quantize+interleave -> uint64 codes
//   2. CUB DeviceSegmentedRadixSort: sort codes (keys) + iota (values) per segment
//   3. invert_perm_segmented_kernel: (B*M) threads, scatter inv_perm from sort_perm

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
__global__ void query_morton_kernel(
    const float* __restrict__ queries,   // (B, M, D) contiguous
    uint64_t* __restrict__ codes,        // (B * M,)
    int B, int M,
    const float* __restrict__ scene_min, // (B, D) contiguous
    const float* __restrict__ scene_max  // (B, D) contiguous
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * M) return;
    const int b = (int)(idx / M);
    const int m = (int)(idx - (int64_t)b * M);
    const float* q  = queries   + ((int64_t)b * M + m) * D;
    const float* lo = scene_min + b * D;
    const float* hi = scene_max + b * D;
    if constexpr (D == 2) {
        codes[idx] = (uint64_t)implicit_bvh::morton::morton_encode_2d(
            q[0], q[1],
            make_float2(lo[0], lo[1]),
            make_float2(hi[0], hi[1]));
    } else {
        codes[idx] = (uint64_t)implicit_bvh::morton::morton_encode_3d(
            q[0], q[1], q[2],
            make_float3(lo[0], lo[1], lo[2]),
            make_float3(hi[0], hi[1], hi[2]));
    }
}


// One thread per flat index: fill out[idx] = idx % M (iota within each segment).
__global__ void fill_iota_segmented_kernel(
    int64_t* __restrict__ out, int64_t M, int64_t total
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = idx % M;
}


// One thread per flat index: scatter inv_perm[b*M + sort_perm[flat]] = i.
// sort_perm[flat] is in [0, M) for each segment b.
__global__ void invert_perm_segmented_kernel(
    const int64_t* __restrict__ sort_perm,
    int64_t* __restrict__ inv_perm,
    int64_t M, int64_t total
) {
    const int64_t flat = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total) return;
    const int64_t b = flat / M;
    const int64_t i = flat - b * M;
    inv_perm[b * M + sort_perm[flat]] = i;
}


template <int D>
std::tuple<torch::Tensor, torch::Tensor> morton_sort_queries_batched_impl(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    c10::cuda::CUDAGuard device_guard(queries.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int B     = (int)queries.size(0);
    const int M     = (int)queries.size(1);
    const int64_t total = (int64_t)B * M;

    auto i64_opts  = queries.options().dtype(torch::kInt64);
    auto byte_opts = queries.options().dtype(torch::kByte);

    // codes and codes_out stored as int64 (uint64 values aliased via reinterpret_cast)
    auto codes     = torch::empty({total}, i64_opts);
    auto codes_out = torch::empty({total}, i64_opts);
    auto values_in = torch::empty({total}, i64_opts);
    auto sort_perm = torch::empty({total}, i64_opts);
    auto inv_perm  = torch::empty({total}, i64_opts);

    constexpr int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    // Kernel 1: Morton codes
    query_morton_kernel<D><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>()),
        B, M,
        scene_min.data_ptr<float>(),
        scene_max.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Kernel 2: iota values [0 .. M-1] per segment
    fill_iota_segmented_kernel<<<blocks, threads, 0, stream>>>(
        values_in.data_ptr<int64_t>(), (int64_t)M, total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // CUB segmented radix sort
    // Segment offsets: [0, M, 2M, ..., B*M] (B+1 elements, int32).
    auto offsets = torch::arange(B + 1, queries.options().dtype(torch::kInt32)) * M;

    // D=3: 30-bit codes, D=2: 32-bit codes. End-bit tells CUB to skip unused high bits.
    constexpr int end_bit = (D == 3) ? 30 : 32;

    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes,
        reinterpret_cast<const uint64_t*>(codes.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(codes_out.data_ptr<int64_t>()),
        values_in.data_ptr<int64_t>(),
        sort_perm.data_ptr<int64_t>(),
        (int)total, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream
    );

    auto temp_storage = torch::empty({(int64_t)(temp_bytes + 1)}, byte_opts);

    cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data_ptr(),
        temp_bytes,
        reinterpret_cast<const uint64_t*>(codes.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(codes_out.data_ptr<int64_t>()),
        values_in.data_ptr<int64_t>(),
        sort_perm.data_ptr<int64_t>(),
        (int)total, B,
        offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1,
        0, end_bit, stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Kernel 3: invert permutation via scatter
    invert_perm_segmented_kernel<<<blocks, threads, 0, stream>>>(
        sort_perm.data_ptr<int64_t>(),
        inv_perm.data_ptr<int64_t>(),
        (int64_t)M, total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(sort_perm.view({B, M}), inv_perm.view({B, M}));
}


std::tuple<torch::Tensor, torch::Tensor> morton_sort_queries_batched_cuda(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    TORCH_CHECK(queries.is_cuda(),           "morton_sort_queries_batched: queries must be a CUDA tensor");
    TORCH_CHECK(queries.is_contiguous(),     "morton_sort_queries_batched: queries must be contiguous");
    TORCH_CHECK(scene_min.is_cuda() && scene_min.is_contiguous(),
                "morton_sort_queries_batched: scene_min must be contiguous CUDA");
    TORCH_CHECK(scene_max.is_cuda() && scene_max.is_contiguous(),
                "morton_sort_queries_batched: scene_max must be contiguous CUDA");
    TORCH_CHECK(queries.scalar_type()   == torch::kFloat32, "morton_sort_queries_batched: queries must be float32");
    TORCH_CHECK(scene_min.scalar_type() == torch::kFloat32, "morton_sort_queries_batched: scene_min must be float32");
    TORCH_CHECK(scene_max.scalar_type() == torch::kFloat32, "morton_sort_queries_batched: scene_max must be float32");
    TORCH_CHECK(queries.dim() == 3,   "morton_sort_queries_batched: queries must have shape (B, M, D)");
    TORCH_CHECK(scene_min.dim() == 2, "morton_sort_queries_batched: scene_min must have shape (B, D)");
    TORCH_CHECK(scene_max.dim() == 2, "morton_sort_queries_batched: scene_max must have shape (B, D)");

    const int D = (int)queries.size(2);
    TORCH_CHECK(D == 2 || D == 3, "morton_sort_queries_batched: D must be 2 or 3, got ", D);
    TORCH_CHECK(queries.size(0) == scene_min.size(0) && queries.size(0) == scene_max.size(0),
                "morton_sort_queries_batched: batch size mismatch");
    TORCH_CHECK((int)scene_min.size(1) == D && (int)scene_max.size(1) == D,
                "morton_sort_queries_batched: scene_min/max last dim must match D");

    if (D == 2) {
        return morton_sort_queries_batched_impl<2>(queries, scene_min, scene_max);
    }
    return morton_sort_queries_batched_impl<3>(queries, scene_min, scene_max);
}
