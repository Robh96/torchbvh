#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>


template <typename scalar_t>
__global__ void smoke_add_one_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] + static_cast<scalar_t>(1);
    }
}


torch::Tensor smoke_add_one_cuda(torch::Tensor input) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    const int64_t numel = input.numel();

    if (numel == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "smoke_add_one_cuda",
        [&] {
            smoke_add_one_kernel<scalar_t><<<
                blocks,
                threads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
