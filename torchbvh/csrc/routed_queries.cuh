#pragma once

template <int D>
__global__ void pack_routed_queries_kernel(
    const float* true_queries, const float* false_queries, const bool* mask,
    float* selected, bool* routes, int total, int queries, int heads) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total) return;
    const int b = row / (queries * heads), within = row - b * queries * heads;
    const int h = within / queries, q = within - h * queries;
    const int source = (b * queries + q) * heads + h;
    const bool route = mask[source];
    routes[row] = route;
    const float* bank = route ? true_queries : false_queries;
    #pragma unroll
    for (int d = 0; d < D; ++d) selected[row * D + d] = bank[source * D + d];
}

template <int D>
__global__ void unpack_routed_gradient_kernel(
    const float* selected, const bool* mask, float* true_gradient, float* false_gradient,
    int total, int queries, int heads) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total) return;
    const int b = row / (queries * heads), within = row - b * queries * heads;
    const int q = within / heads, h = within - q * heads;
    const int source = (b * heads + h) * queries + q;
    const bool route = mask[row];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        const float value = selected[source * D + d];
        if (true_gradient) true_gradient[row * D + d] = route ? value : 0;
        if (false_gradient) false_gradient[row * D + d] = route ? 0 : value;
    }
}

std::tuple<torch::Tensor, torch::Tensor> pack_routed_queries_cuda(
    torch::Tensor true_queries, torch::Tensor false_queries, torch::Tensor mask) {
    TORCH_CHECK(true_queries.is_cuda() && true_queries.is_contiguous() && true_queries.scalar_type() == torch::kFloat32 &&
                true_queries.dim() == 4 && (true_queries.size(3) == 2 || true_queries.size(3) == 3), "pack_queries: invalid query tensor");
    TORCH_CHECK(false_queries.is_cuda() && false_queries.is_contiguous() && false_queries.scalar_type() == torch::kFloat32 &&
                false_queries.device() == true_queries.device() && false_queries.sizes() == true_queries.sizes(), "pack_queries: incompatible false queries");
    const int B = true_queries.size(0), M = true_queries.size(1), H = true_queries.size(2), D = true_queries.size(3);
    TORCH_CHECK(B > 0 && M > 0 && H > 0 && static_cast<int64_t>(B) * M * H <= INT32_MAX &&
                mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.device() == true_queries.device() &&
                mask.dim() == 3 && mask.size(0) == B && mask.size(1) == M && mask.size(2) == H, "pack_queries: invalid mask or extent");
    c10::cuda::CUDAGuard guard(true_queries.device());
    auto selected = torch::empty({B,H,M,D}, true_queries.options());
    auto routes = torch::empty({B,H*M}, mask.options());
    const int blocks = (B*H*M + 255) / 256;
#define PACK(DIM) pack_routed_queries_kernel<DIM><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(true_queries.data_ptr<float>(), false_queries.data_ptr<float>(),mask.data_ptr<bool>(),selected.data_ptr<float>(),routes.data_ptr<bool>(),B*H*M,M,H)
    if (D == 2) { PACK(2); } else { PACK(3); }
#undef PACK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected,routes};
}

std::tuple<torch::Tensor, torch::Tensor> unpack_routed_gradient_cuda(
    torch::Tensor selected, torch::Tensor mask, bool need_true, bool need_false) {
    TORCH_CHECK(selected.is_cuda() && selected.is_contiguous() && selected.scalar_type() == torch::kFloat32 &&
                selected.dim() == 4 && (selected.size(3) == 2 || selected.size(3) == 3), "unpack_queries: invalid gradient tensor");
    const int B=selected.size(0), H=selected.size(1), M=selected.size(2), D=selected.size(3);
    TORCH_CHECK(mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.device() == selected.device() &&
                mask.dim() == 3 && mask.size(0) == B && mask.size(1) == M && mask.size(2) == H, "unpack_queries: invalid mask");
    c10::cuda::CUDAGuard guard(selected.device());
    auto true_gradient = torch::empty(need_true ? std::vector<int64_t>{B,M,H,D} : std::vector<int64_t>{0}, selected.options());
    auto false_gradient = torch::empty(need_false ? std::vector<int64_t>{B,M,H,D} : std::vector<int64_t>{0}, selected.options());
    const int blocks = (B*H*M + 255) / 256;
#define UNPACK(DIM) unpack_routed_gradient_kernel<DIM><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(selected.data_ptr<float>(),mask.data_ptr<bool>(),need_true ? true_gradient.data_ptr<float>() : nullptr,need_false ? false_gradient.data_ptr<float>() : nullptr,B*H*M,M,H)
    if (D == 2) { UNPACK(2); } else { UNPACK(3); }
#undef UNPACK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {true_gradient,false_gradient};
}
