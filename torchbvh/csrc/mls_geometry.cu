// Geometry-owner MLS: one two-lane team per query, native feature banks.
// Keep the per-channel solves and ordered FP32 query-gradient accumulation.
// Cholesky storage is triangular, followed by bandwidth: 7 (2D), 11 (3D).
#include <type_traits>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cmath>
#include <tuple>

namespace geometry_owner {
constexpr int kNeighbors = 4;
constexpr int kTeamLanes = 2;
constexpr int kThreads = 128;

template <int D> struct SavedGeometry {
    static constexpr int basis_size = D + 1;
    static constexpr int bandwidth_offset = basis_size * (basis_size + 1) / 2;
    static constexpr int width = bandwidth_offset + 1;
};

template <int D>
__device__ __forceinline__ float packed_basis_value(
    const float* __restrict__ delta,
    int j,
    int i
) {
    return i == 0 ? 1.0f : delta[j * D + i - 1];
}

// Preserve the solve's arithmetic while accessing native (B,N,H,C) banks.
__device__ __forceinline__ float native_feature_value(
    const float* features, const float* false_features, int64_t fb,
    int64_t index, int source_count, int channels, int heads, int true_count, int c) {
    const bool route = index < true_count;
    const int count = route ? true_count : source_count - true_count;
    const int64_t local = route ? index : index - true_count;
    const float* bank = route ? features : false_features;
    return bank[((fb / heads * count + local) * heads + fb % heads) * channels + c];
}

__device__ __forceinline__ float* native_feature_gradient(
    float* gradient, int64_t src, int source_count, int channels, int heads, int c) {
    const int64_t fb = src / source_count;
    const int64_t index = src - fb * source_count;
    return gradient + ((fb / heads * source_count + index) * heads + fb % heads) * channels + c;
}


template <int D, bool ReturnGradient>
__global__ void forward_kernel(
    const float* __restrict__ displaced_points,
    const int64_t* __restrict__ indices,
    const float* __restrict__ squared_distances,
    const float* __restrict__ features,
    const float* __restrict__ source_points,
    const int64_t* __restrict__ query_order,
    float* __restrict__ interpolated,
    float* __restrict__ field_gradient,
    float* __restrict__ factors,
    int32_t* __restrict__ exact_counts,
    int total_queries,
    int feature_batches,
    int source_count,
    int channels,
    int queries_per_batch,
    int queries_per_head,
    float regularization,
    float bandwidth_min,
    float exact_eps,
    const float* false_features, int true_count, bool query_major_output = false
) {
    constexpr int P = D + 1;
    constexpr int K = kNeighbors;
    constexpr int ChannelTile = kTeamLanes;
    constexpr int FS = SavedGeometry<D>::width;
    const unsigned team_mask = __activemask();
    const int storage_qrow = (int)(blockIdx.x * blockDim.x + threadIdx.x) / ChannelTile;
    const int channel_lane = threadIdx.x % ChannelTile;
    if (storage_qrow >= total_queries) return;
    const int sample = storage_qrow / queries_per_batch;
    const int qrow = sample * queries_per_batch + (int)query_order[storage_qrow];
    const int64_t fb = (int64_t)sample * (queries_per_batch / queries_per_head)
            + ((qrow - sample * queries_per_batch) / queries_per_head);
    const int output_qrow = query_major_output
        ? sample * queries_per_batch + ((qrow - sample * queries_per_batch) % queries_per_head)
            * (queries_per_batch / queries_per_head) + (qrow - sample * queries_per_batch) / queries_per_head
        : qrow;
    if (fb < 0 || fb >= feature_batches) return;
    float bandwidth = fmaxf(
        squared_distances[storage_qrow * K + ((K - 1) / 2)], bandwidth_min);

    float delta[K * D];
    float weights[K];
    int exact_count = 0;
    int exact_mask = 0;
    if (channel_lane == 0) {
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
            delta[j * D + d] = dv;
            dsq += dv * dv;
        }
        weights[j] = expf(-dsq / (2.0f * bandwidth));
        const bool hit = squared_distances[storage_qrow * K + j] <= exact_eps;
        exact_count += hit; exact_mask |= int(hit) << j;
    }
    }
    if constexpr (ChannelTile > 1) {
        exact_count = __shfl_sync(team_mask, exact_count, 0, ChannelTile);
        exact_mask = __shfl_sync(team_mask, exact_mask, 0, ChannelTile);
        #pragma unroll
        for (int i = 0; i < K * D; ++i) delta[i] = __shfl_sync(team_mask, delta[i], 0, ChannelTile);
        #pragma unroll
        for (int i = 0; i < K; ++i) weights[i] = __shfl_sync(team_mask, weights[i], 0, ChannelTile);
    }
    if (channel_lane == 0) {
        exact_counts[storage_qrow] = exact_mask;
        factors[storage_qrow * FS + SavedGeometry<D>::bandwidth_offset] = bandwidth;
    }

    if (exact_count > 0) {
        for (int c = channel_lane; c < channels; c += ChannelTile) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if (squared_distances[storage_qrow * K + j] <= exact_eps) {
                    sum += native_feature_value(features, false_features, fb, indices[storage_qrow * K + j], source_count, channels, queries_per_batch / queries_per_head, true_count, c);
                }
            }
            interpolated[output_qrow * channels + c] = sum / (float)exact_count;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                if constexpr (ReturnGradient) field_gradient[(output_qrow * D + d) * channels + c] = 0.0f;
            }
        }
        return;
    }

    float L[P * P];
    if (channel_lane == 0) {
    float A[P * P];
    #pragma unroll
    for (int i = 0; i < P; ++i) {
        #pragma unroll
        for (int l = 0; l < P; ++l) {
            float s = i == l ? regularization : 0.0f;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                s += weights[j] * packed_basis_value<D>(delta, j, i)
                    * packed_basis_value<D>(delta, j, l);
            }
            A[i * P + l] = s;
        }
    }

    float trace = 0.0f;
    #pragma unroll
    for (int i = 0; i < P; ++i) trace += A[i * P + i];
    const float jitter = fmaxf(1.0e-5f, 1.0e-4f * trace / (float)P);
    bool solve_stable = false;
    bool used_jitter = false;
    #pragma unroll
    for (int attempt = 0; attempt < 2; ++attempt) {
        used_jitter |= attempt != 0;
        int stable = 1;
        const float diagonal_jitter = attempt == 0 ? 0.0f : jitter;
        #pragma unroll
        for (int idx = 0; idx < P * P; ++idx) L[idx] = 0.0f;
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            #pragma unroll
            for (int j = 0; j <= i; ++j) {
                float s = A[i * P + j] + (i == j ? diagonal_jitter : 0.0f);
                #pragma unroll
                for (int kk = 0; kk < P; ++kk) {
                    if (kk < j) s -= L[i * P + kk] * L[j * P + kk];
                }
                if (i == j) {
                    const float scale = fabsf(A[i * P + i] + diagonal_jitter);
                    const float floor = fmaxf(1.0e-12f, fminf(1.0e-5f, 1.0e-3f * scale));
                    if (!(s > floor)) { stable = 0; s = floor; }
                    if (channel_lane == 0) factors[storage_qrow * FS + (i * (i + 1) / 2 + i)] = sqrtf(s);
                    L[i * P + i] = sqrtf(s);
                } else {
                    const float denom = L[j * P + j];
                    const float value = denom > 0.0f ? s / denom : 0.0f;
                    stable &= denom > 0.0f;
                    L[i * P + j] = value;
                    if (channel_lane == 0) factors[storage_qrow * FS + (i * (i + 1) / 2 + j)] = value;
                }
            }
        }
        if (stable) { solve_stable = true; break; }
    }

    }
    if constexpr (ChannelTile > 1) {
        #pragma unroll
        for (int i = 0; i < P * P; ++i) L[i] = __shfl_sync(team_mask, L[i], 0, ChannelTile);
    }
    for (int c = channel_lane; c < channels; c += ChannelTile) {
        float rhs[P] = {};
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float f = native_feature_value(features, false_features, fb, indices[storage_qrow * K + j], source_count, channels, queries_per_batch / queries_per_head, true_count, c);
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                rhs[i] += weights[j] * packed_basis_value<D>(delta, j, i) * f;
            }
        }
        float y[P], x[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = rhs[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j < i) s -= L[i * P + j] * y[j];
            y[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = y[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j > i) s -= L[j * P + i] * x[j];
            x[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }
        interpolated[output_qrow * channels + c] = x[0];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            if constexpr (ReturnGradient) field_gradient[(output_qrow * D + d) * channels + c] = x[d + 1];
        }
    }
}

template <int D, bool NeedQuery, bool NeedFeatures>
__global__ void backward_kernel(
    const float* __restrict__ displaced_points,
    const int64_t* __restrict__ indices,
    const float* __restrict__ squared_distances,
    const float* __restrict__ features,
    const float* __restrict__ source_points,
    const int64_t* __restrict__ query_order,
    const float* __restrict__ factors,
    const int32_t* __restrict__ exact_counts,
    const float* __restrict__ d_interpolated,
    const float* __restrict__ d_field_gradient,
    float* __restrict__ d_features,
    float* __restrict__ d_displaced_points,
    int total_queries,
    int feature_batches,
    int source_count,
    int channels,
    int queries_per_batch,
    int queries_per_head,
    float bandwidth_min,
    float exact_eps,
    const float* false_features, int true_count, bool query_major_output = false
) {
    constexpr int P = D + 1;
    constexpr int K = kNeighbors;
    constexpr int ChannelTile = kTeamLanes;
    constexpr int FS = SavedGeometry<D>::width;
    const unsigned team_mask = __activemask();
    const int storage_qrow = (int)(blockIdx.x * blockDim.x + threadIdx.x) / ChannelTile;
    const int channel_lane = threadIdx.x % ChannelTile;
    if (storage_qrow >= total_queries) return;
    const int sample = storage_qrow / queries_per_batch;
    const int qrow = sample * queries_per_batch + (int)query_order[storage_qrow];
    const int64_t fb = (int64_t)sample * (queries_per_batch / queries_per_head)
            + ((qrow - sample * queries_per_batch) / queries_per_head);
    const int output_qrow = query_major_output
        ? sample * queries_per_batch + ((qrow - sample * queries_per_batch) % queries_per_head)
            * (queries_per_batch / queries_per_head) + (qrow - sample * queries_per_batch) / queries_per_head
        : qrow;
    if (fb < 0 || fb >= feature_batches) {
        #pragma unroll
        for (int d = 0; d < D; ++d) if constexpr (NeedQuery) if (channel_lane == 0) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }
    const int exact_mask = exact_counts[storage_qrow];
    const int exact_count = __popc(exact_mask);
    if (exact_count > 0) {
        for (int c = channel_lane; c < channels; c += ChannelTile) {
            const float scale = d_interpolated[output_qrow * channels + c] / (float)exact_count;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if ((exact_mask & (1 << j)) != 0) {
                    if constexpr (NeedFeatures) atomicAdd(native_feature_gradient(d_features, fb * source_count + indices[storage_qrow * K + j], source_count, channels, queries_per_batch / queries_per_head, c), scale);
                }
            }
        }
        #pragma unroll
        for (int d = 0; d < D; ++d) if constexpr (NeedQuery) if (channel_lane == 0) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    float L[P * P];
    int ill_conditioned = 0;
    if (channel_lane == 0) {
    #pragma unroll
    for (int i = 0; i < P; ++i) {
        #pragma unroll
        for (int j = 0; j < P; ++j) {
            const float stored = j > i ? 0.f : factors[storage_qrow * FS + (i * (i + 1) / 2 + j)];
            if (j > i) L[i * P + j] = 0.0f;
            else if (i == j) {
                ill_conditioned |= stored <= 0.0f;
                L[i * P + i] = stored;
            } else L[i * P + j] = stored;
        }
    }
    }
    if constexpr (ChannelTile > 1) {
        ill_conditioned = __shfl_sync(team_mask, ill_conditioned, 0, ChannelTile);
        #pragma unroll
        for (int i = 0; i < P * P; ++i) L[i] = __shfl_sync(team_mask, L[i], 0, ChannelTile);
    }
    if (ill_conditioned) {
        #pragma unroll
        for (int d = 0; d < D; ++d) if constexpr (NeedQuery) if (channel_lane == 0) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    const float bandwidth = fmaxf(
        factors[storage_qrow * FS + SavedGeometry<D>::bandwidth_offset], bandwidth_min);
    float delta[K * D];
    float weights[K];
    if (channel_lane == 0) {
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
            delta[j * D + d] = dv;
            dsq += dv * dv;
        }
        weights[j] = expf(-dsq / (2.0f * bandwidth));
    }

    }
    if constexpr (ChannelTile > 1) {
        #pragma unroll
        for (int i = 0; i < K * D; ++i) delta[i] = __shfl_sync(team_mask, delta[i], 0, ChannelTile);
        #pragma unroll
        for (int i = 0; i < K; ++i) weights[i] = __shfl_sync(team_mask, weights[i], 0, ChannelTile);
    }
    float dq[D] = {};
    for (int c = channel_lane; c < channels; c += ChannelTile) {
        float dx[P];
        dx[0] = d_interpolated[output_qrow * channels + c];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dx[d + 1] = d_field_gradient == nullptr
                ? 0.0f : d_field_gradient[(output_qrow * D + d) * channels + c];
        }
        float yg[P], G[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = dx[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j < i) s -= L[i * P + j] * yg[j];
            yg[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = yg[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j > i) s -= L[j * P + i] * G[j];
            G[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }

        float x[P] = {};
        if constexpr (NeedQuery) {
        float rhs[P] = {};
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float f = native_feature_value(features, false_features, fb, indices[storage_qrow * K + j], source_count, channels, queries_per_batch / queries_per_head, true_count, c);
            #pragma unroll
            for (int i = 0; i < P; ++i) rhs[i] += weights[j] * packed_basis_value<D>(delta, j, i) * f;
        }
        float yx[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = rhs[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j < i) s -= L[i * P + j] * yx[j];
            yx[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = yx[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j > i) s -= L[j * P + i] * x[j];
            x[i] = s / fmaxf(L[i * P + i], 1.0e-20f);
        }

        }

        float channel_terms[K * D];
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const int64_t src = fb * source_count + indices[storage_qrow * K + j];
            float f = 0.f;
            if constexpr (NeedQuery) f = native_feature_value(features, false_features, fb, src - fb * source_count, source_count, channels, queries_per_batch / queries_per_head, true_count, c);
            float df = 0.0f, ay = 0.0f, fitted = 0.0f;
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                const float phi = packed_basis_value<D>(delta, j, i);
                if constexpr (NeedFeatures) df += weights[j] * phi * G[i];
                if constexpr (NeedQuery) ay += phi * G[i];
                if constexpr (NeedQuery) fitted += phi * x[i];
            }
            if constexpr (NeedFeatures) atomicAdd(native_feature_gradient(d_features, src, source_count, channels, queries_per_batch / queries_per_head, c), df);
            if constexpr (NeedQuery) {
            const float residual = f - fitted;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                const float term = weights[j] * (residual * G[d + 1] - ay * x[d + 1])
                    - ay * residual * weights[j] * delta[j * D + d] / bandwidth;
                if constexpr (ChannelTile == 1) dq[d] += term;
                else channel_terms[j * D + d] = term;
            }
            }
        }
        if constexpr (ChannelTile > 1 && NeedQuery) {
            const unsigned active = __activemask();
            // Accumulate in the original channel/neighbor order, including
            // each intermediate float32 rounding, rather than a tree sum.
            #pragma unroll
            for (int lane = 0; lane < ChannelTile; ++lane) {
                #pragma unroll
                for (int j = 0; j < K; ++j) {
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        const float term = __shfl_sync(active, channel_terms[j * D + d], lane, ChannelTile);
                        if (channel_lane == 0) dq[d] += term;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        if constexpr (NeedQuery) if (channel_lane == 0) d_displaced_points[qrow * D + d] = dq[d];
    }
}




template <int D, bool Slopes>
std::vector<torch::Tensor> launch_forward(
    torch::Tensor queries, torch::Tensor sources, torch::Tensor indices,
    torch::Tensor distances, torch::Tensor features, torch::Tensor false_features,
    torch::Tensor order, int per_batch, int per_head, float regularization,
    float bandwidth_min, float exact_epsilon, bool query_major) {
    const int count = queries.size(0), channels = features.size(3);
    torch::Tensor values = torch::empty({count, channels}, features.options());
    torch::Tensor slopes = Slopes ? torch::empty({count,D,channels},features.options())
                         : torch::empty({0},features.options());
    torch::Tensor state = torch::empty({count,SavedGeometry<D>::width},features.options());
    torch::Tensor hits = torch::empty({count},indices.options().dtype(torch::kInt32));
    const int blocks = (int64_t(count)*kTeamLanes+kThreads-1)/kThreads;
    if (count) forward_kernel<D,Slopes><<<blocks,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
        queries.data_ptr<float>(),indices.data_ptr<int64_t>(),distances.data_ptr<float>(),
        features.data_ptr<float>(),sources.data_ptr<float>(),order.data_ptr<int64_t>(),
        values.data_ptr<float>(),slopes.data_ptr<float>(),state.data_ptr<float>(),hits.data_ptr<int32_t>(),
        count,features.size(0)*features.size(2),sources.size(1),channels,per_batch,per_head,
        regularization,bandwidth_min,exact_epsilon,false_features.data_ptr<float>(),features.size(1),query_major);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {values,slopes,state,hits};
}

template <int D, bool NeedQueries, bool NeedFeatures>
std::vector<torch::Tensor> launch_backward(
    torch::Tensor queries, torch::Tensor sources, torch::Tensor indices,
    torch::Tensor features, torch::Tensor false_features, torch::Tensor order,
    torch::Tensor state, torch::Tensor hits, torch::Tensor value_gradients,
    torch::Tensor slope_gradients, int per_batch, int per_head,
    float bandwidth_min, float exact_epsilon, bool query_major) {
    const int count = queries.size(0), channels = features.size(3);
    auto feature_gradients = NeedFeatures ? torch::zeros(
        {features.size(0),sources.size(1),features.size(2),channels},features.options())
        : torch::empty({0},features.options());
    auto query_gradients = NeedQueries ? torch::empty_like(queries) : torch::empty({0},queries.options());
    const int blocks = (int64_t(count)*kTeamLanes+kThreads-1)/kThreads;
    if (count) backward_kernel<D,NeedQueries,NeedFeatures><<<blocks,kThreads,0,at::cuda::getCurrentCUDAStream()>>>(
        queries.data_ptr<float>(),indices.data_ptr<int64_t>(),nullptr,features.data_ptr<float>(),
        sources.data_ptr<float>(),order.data_ptr<int64_t>(),state.data_ptr<float>(),hits.data_ptr<int32_t>(),
        value_gradients.data_ptr<float>(),slope_gradients.numel()?slope_gradients.data_ptr<float>():nullptr,
        feature_gradients.data_ptr<float>(),query_gradients.data_ptr<float>(),count,
        features.size(0)*features.size(2),sources.size(1),channels,per_batch,per_head,
        bandwidth_min,exact_epsilon,false_features.data_ptr<float>(),features.size(1),query_major);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {feature_gradients,query_gradients};
}

void validate_inputs(torch::Tensor queries, torch::Tensor sources, torch::Tensor indices,
    torch::Tensor features, torch::Tensor false_features, torch::Tensor order, int per_batch, int per_head) {
    for (const auto& tensor : {queries,sources,features,false_features}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type()==torch::kFloat32 && tensor.is_contiguous(),
                    "geometry MLS: contiguous float32 CUDA inputs required");
        TORCH_CHECK(tensor.device()==queries.device(),"geometry MLS: devices must match");
    }
    for (const auto& tensor : {indices,order}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type()==torch::kInt64 && tensor.is_contiguous(),
                    "geometry MLS: contiguous int64 CUDA indices/order required");
        TORCH_CHECK(tensor.device()==queries.device(),"geometry MLS: devices must match");
    }
    TORCH_CHECK(queries.dim()==2 && (queries.size(1)==2 || queries.size(1)==3),"geometry MLS: D must be 2/3");
    TORCH_CHECK(queries.size(0)<=INT32_MAX/2,"geometry MLS: query extent exceeds team indexing");
    TORCH_CHECK(sources.dim()==3 && sources.size(2)==queries.size(1),"geometry MLS: invalid source layout");
    TORCH_CHECK(indices.dim()==2 && indices.size(0)==queries.size(0) && indices.size(1)==4,
                "geometry MLS: expected (Q,4) indices");
    TORCH_CHECK(order.dim()==1 && order.numel()==queries.size(0),"geometry MLS: invalid query order");
    TORCH_CHECK(features.dim()==4 && false_features.dim()==4 && features.size(0)==sources.size(0),
                "geometry MLS: expected native (B,N,H,C) feature banks");
    TORCH_CHECK(false_features.size(0)==features.size(0) && false_features.size(2)==features.size(2)
        && false_features.size(3)==features.size(3) && sources.size(1)==features.size(1)+false_features.size(1),
        "geometry MLS: incompatible feature banks");
    const auto channels=features.size(3);
    TORCH_CHECK(channels==4 || channels==8 || channels==16 || channels==32 || channels==64,
                "geometry MLS: unsupported channel count");
    TORCH_CHECK(per_head>0 && per_batch>0 && per_batch%per_head==0
        && per_batch/per_head==features.size(2) && int64_t(per_batch)*sources.size(0)==queries.size(0),
        "geometry MLS: invalid query layout");
}
} // namespace geometry_owner

std::vector<torch::Tensor> mls_geometry_forward_cuda(
    torch::Tensor queries, torch::Tensor sources, torch::Tensor indices, torch::Tensor distances,
    torch::Tensor features, torch::Tensor false_features, torch::Tensor order, int per_batch, int per_head,
    double regularization, double bandwidth_min, double exact_epsilon, bool slopes, bool query_major) {
    geometry_owner::validate_inputs(queries,sources,indices,features,false_features,order,per_batch,per_head);
    TORCH_CHECK(distances.is_cuda() && distances.device()==queries.device()
        && distances.scalar_type()==torch::kFloat32 && distances.is_contiguous()
        && distances.sizes()==indices.sizes(),"geometry MLS: invalid squared distances");
    c10::cuda::CUDAGuard guard(queries.device());
#define FORWARD(D) return slopes ? geometry_owner::launch_forward<D,true>(queries,sources,indices,distances,features,false_features,order,per_batch,per_head,regularization,bandwidth_min,exact_epsilon,query_major) : geometry_owner::launch_forward<D,false>(queries,sources,indices,distances,features,false_features,order,per_batch,per_head,regularization,bandwidth_min,exact_epsilon,query_major)
    if (queries.size(1)==2) { FORWARD(2); }
    FORWARD(3);
#undef FORWARD
}

std::vector<torch::Tensor> mls_geometry_backward_cuda(
    torch::Tensor queries, torch::Tensor sources, torch::Tensor indices, torch::Tensor distances,
    torch::Tensor features, torch::Tensor false_features, torch::Tensor order, torch::Tensor state,
    torch::Tensor hits, torch::Tensor value_gradients, torch::Tensor slope_gradients,
    int per_batch, int per_head, double bandwidth_min, double exact_epsilon,
    bool query_major, bool need_queries, bool need_features) {
    geometry_owner::validate_inputs(queries,sources,indices,features,false_features,order,per_batch,per_head);
    c10::cuda::CUDAGuard guard(queries.device());
    const int state_width=queries.size(1)==2?7:11;
    TORCH_CHECK(state.is_contiguous() && state.is_cuda() && state.device()==queries.device()
        && state.scalar_type()==torch::kFloat32 && state.dim()==2 && state.size(0)==queries.size(0)
        && state.size(1)==state_width,"geometry MLS: incompatible saved state");
    TORCH_CHECK(hits.is_contiguous() && hits.is_cuda() && hits.device()==queries.device()
        && hits.scalar_type()==torch::kInt32 && hits.numel()==queries.size(0),"geometry MLS: invalid hit mask");
    TORCH_CHECK(value_gradients.is_cuda() && value_gradients.device()==queries.device()
        && value_gradients.scalar_type()==torch::kFloat32 && value_gradients.is_contiguous()
        && value_gradients.dim()==2 && value_gradients.size(0)==queries.size(0)
        && value_gradients.size(1)==features.size(3),"geometry MLS: invalid value gradients");
    TORCH_CHECK(slope_gradients.is_cuda() && slope_gradients.device()==queries.device()
        && slope_gradients.scalar_type()==torch::kFloat32 && slope_gradients.is_contiguous()
        && (!slope_gradients.numel() || (slope_gradients.dim()==3 && slope_gradients.size(0)==queries.size(0)
        && slope_gradients.size(1)==queries.size(1) && slope_gradients.size(2)==features.size(3))),
        "geometry MLS: invalid slope gradients");
    if (!need_queries && !need_features) return {torch::empty({0},features.options()),torch::empty({0},queries.options())};
#define BACKWARD(D) do { \
    if (!need_queries) return geometry_owner::launch_backward<D,false,true>(queries,sources,indices,features,false_features,order,state,hits,value_gradients,slope_gradients,per_batch,per_head,bandwidth_min,exact_epsilon,query_major); \
    if (!need_features) return geometry_owner::launch_backward<D,true,false>(queries,sources,indices,features,false_features,order,state,hits,value_gradients,slope_gradients,per_batch,per_head,bandwidth_min,exact_epsilon,query_major); \
    return geometry_owner::launch_backward<D,true,true>(queries,sources,indices,features,false_features,order,state,hits,value_gradients,slope_gradients,per_batch,per_head,bandwidth_min,exact_epsilon,query_major); \
} while(false)
    if (queries.size(1)==2) { BACKWARD(2); }
    BACKWARD(3);
#undef BACKWARD
}
