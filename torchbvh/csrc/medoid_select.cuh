// Small fixed-population medoids. Source Morton ranks and two-member ties are
// unchanged. Explicit rounded operations match the unfused PyTorch arithmetic.
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>

template <int D>
__global__ void select_bvh_medoids_kernel(
    const float* points, const int64_t* sorted, const int64_t* ranks,
    const bool* valid, const int64_t* counts, int64_t* medoids,
    float* centroids, int batch, int count, int groups, int population) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch * groups) return;
    const int b = row / groups, group = row % groups;
    float selected[4 * D] = {}, center[D] = {};
    int64_t members[4];
    for (int j = 0; j < population; ++j) {
        members[j] = sorted[(int64_t)b * count + ranks[group * population + j]];
        for (int d = 0; d < D; ++d) {
            const float value = points[((int64_t)b * count + members[j]) * D + d];
            selected[j * D + d] = value;
            center[d] = __fadd_rn(center[d], valid[group * population + j] ? value : 0.f);
        }
    }
    for (int d = 0; d < D; ++d) {
        center[d] = __fdiv_rn(center[d], (float)counts[group]);
        centroids[row * D + d] = center[d];
    }
    float best = CUDART_INF_F;
    int slot = 0;
    for (int j = 0; j < population; ++j) {
        if (!valid[group * population + j]) continue;
        float distance = 0.f;
        for (int d = 0; d < D; ++d) {
            const float delta = __fsub_rn(selected[j * D + d], center[d]);
            distance = __fadd_rn(distance, __fmul_rn(delta, delta));
        }
        if (distance < best) { best = distance; slot = j; }
    }
    medoids[row] = members[counts[group] == 2 ? 0 : slot];
}

std::tuple<torch::Tensor, torch::Tensor> select_bvh_medoids_cuda(
    torch::Tensor points, torch::Tensor sorted, torch::Tensor ranks,
    torch::Tensor valid, torch::Tensor counts) {
    TORCH_CHECK(points.is_cuda() && points.is_contiguous() && points.scalar_type() == torch::kFloat32
                && points.dim() == 3, "select_bvh_medoids: contiguous float32 CUDA points required");
    for (const auto& tensor : {sorted, ranks, counts})
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.device() == points.device()
                    && tensor.scalar_type() == torch::kInt64, "select_bvh_medoids: contiguous same-device int64 required");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.device() == points.device()
                && valid.scalar_type() == torch::kBool, "select_bvh_medoids: contiguous same-device bool required");
    TORCH_CHECK(ranks.dim() == 2 && ranks.size(0) > 0 && ranks.size(1) > 0 && ranks.size(1) <= 4
                && valid.sizes() == ranks.sizes() && counts.dim() == 1 && counts.size(0) == ranks.size(0)
                && sorted.dim() == 2 && sorted.size(0) == points.size(0) && sorted.size(1) == points.size(1),
                "select_bvh_medoids: incompatible template shapes (maximum population four)");
    TORCH_CHECK(points.size(0) > 0 && points.size(1) > 0 && points.size(0) * points.size(1) <= INT_MAX
                && points.size(0) * ranks.size(0) <= INT_MAX && (points.size(2) == 2 || points.size(2) == 3),
                "select_bvh_medoids: invalid dimensions or indexing extent");
    c10::cuda::CUDAGuard guard(points.device());
    auto medoids = torch::empty({points.size(0), ranks.size(0)}, sorted.options());
    auto centroids = torch::empty({points.size(0), ranks.size(0), points.size(2)}, points.options());
    const int blocks = (points.size(0) * ranks.size(0) + 127) / 128;
#define SELECT_MEDOIDS(D) select_bvh_medoids_kernel<D><<<blocks,128,0,at::cuda::getCurrentCUDAStream()>>>(points.data_ptr<float>(),sorted.data_ptr<int64_t>(),ranks.data_ptr<int64_t>(),valid.data_ptr<bool>(),counts.data_ptr<int64_t>(),medoids.data_ptr<int64_t>(),centroids.data_ptr<float>(),points.size(0),points.size(1),ranks.size(0),ranks.size(1))
    if (points.size(2) == 2) { SELECT_MEDOIDS(2); } else { SELECT_MEDOIDS(3); }
#undef SELECT_MEDOIDS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {medoids,centroids};
}
