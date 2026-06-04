#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cmath>
#include <tuple>

// Cooperative fused MLS forward kernel.
//
// Thread layout per query block:
//   Phase 1 (threads 0..K-1): compute basis and weighted basis in parallel.
//   Phase 2 (threads 0..P-1): compute one normal-matrix row each in parallel.
//   Phase 3 (thread 0):       Cholesky factorization of the P×P SPD normal matrix.
//   Phase 4 (all threads):    RHS construction + triangular solve, parallelised over C.
//
// Normal matrix: A[i,l] = sum_j wb[j,i]*basis[j,l] + λ*δ(i,l)
//   where wb[j,i] = w_j * φ_j_i, basis[j,i] = φ_j_i, w_j = exp(-|Δj|²/2h).
// This matches the reference Python computation exactly, preserving float32 ordering.
// A is symmetric positive definite; Cholesky is unconditionally stable with λ > 0.
// Factors output stores L (lower-triangular) for Python diagonal-check fallback.

template <int D, int K>
__global__ void mls_fused_forward_kernel(
    const float* __restrict__ displaced_points,   // (M, D)
    const float* __restrict__ neighbor_positions, // (M, K, D)
    const int64_t* __restrict__ indices,          // (M, K)
    const float* __restrict__ squared_distances,  // (M, K)
    const float* __restrict__ features,           // (B, N, C)
    const int64_t* __restrict__ feature_batch,    // (M,)
    float* __restrict__ interpolated,             // (M, C)
    float* __restrict__ field_gradient,           // (M, D, C)
    float* __restrict__ factors,                  // (M, P, P) — Cholesky L, lower-triangular
    int32_t* __restrict__ exact_counts,           // (M,)
    int total_queries,
    int feature_batches,
    int source_count,
    int channels,
    float regularization,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int qrow = blockIdx.x;
    if (qrow >= total_queries) return;

    __shared__ float   bw_sh;
    __shared__ int64_t fb_sh;
    __shared__ int     exact_sh;
    __shared__ float   basis_sh[K * P];   // φ_j_i  (unweighted)
    __shared__ float   wb_sh[K * P];      // w_j * φ_j_i  (matches reference numerics)
    __shared__ float   A_sh[P * P];       // normal matrix (written by P threads)
    __shared__ float   L_sh[P * P];       // Cholesky factor (written by thread 0)

    // ---- Phase 0: scalars (thread 0 only) ----
    if (threadIdx.x == 0) {
        fb_sh    = (feature_batch != nullptr) ? feature_batch[qrow] : 0LL;
        bw_sh    = fmaxf(squared_distances[qrow * K + ((K - 1) / 2)], bandwidth_min);
        exact_sh = 0;
    }
    __syncthreads();

    // ---- Phase 1: basis and weighted basis (threads 0..K-1 in parallel) ----
    if ((int)threadIdx.x < K) {
        const int j = (int)threadIdx.x;
        float delta[D];
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const float dv = displaced_points[qrow * D + d]
                           - neighbor_positions[(qrow * K + j) * D + d];
            delta[d] = dv;
            dsq     += dv * dv;
        }
        const float w    = expf(-dsq / (2.0f * bw_sh));
        basis_sh[j * P]  = 1.0f;
        wb_sh[j * P]     = w;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            basis_sh[j * P + 1 + d] = delta[d];
            wb_sh[j * P + 1 + d]    = w * delta[d];
        }
        if (squared_distances[qrow * K + j] <= exact_eps) atomicAdd(&exact_sh, 1);
    }
    __syncthreads();

    // ---- Phase 2: normal matrix rows (threads 0..P-1 in parallel) ----
    // A[i,l] = sum_j wb[j,i]*basis[j,l] + λ*δ(i,l)
    if ((int)threadIdx.x < P) {
        const int i = (int)threadIdx.x;
        #pragma unroll
        for (int l = 0; l < P; ++l) {
            float s = (i == l) ? regularization : 0.0f;
            #pragma unroll
            for (int j = 0; j < K; ++j) s += wb_sh[j * P + i] * basis_sh[j * P + l];
            A_sh[i * P + l] = s;
        }
    }
    __syncthreads();

    // ---- Phase 3: Cholesky factorization (thread 0) ----
    // factors stores lower off-diagonal L entries and diagonal raw Cholesky
    // residuals s, not sqrt(s), so backward can reconstruct L.
    if (threadIdx.x == 0) {
        float trace = 0.0f;
        #pragma unroll
        for (int i = 0; i < P; ++i) trace += A_sh[i * P + i];
        const float jitter = fmaxf(1.0e-5f, 1.0e-4f * trace / static_cast<float>(P));

        int stable = 0;
        #pragma unroll
        for (int attempt = 0; attempt < 2; ++attempt) {
            stable = 1;
            const float diagonal_jitter = (attempt == 0) ? 0.0f : jitter;

            #pragma unroll
            for (int idx = 0; idx < P * P; ++idx) {
                L_sh[idx] = 0.0f;
            }

            #pragma unroll
            for (int i = 0; i < P; ++i) {
                #pragma unroll
                for (int j = 0; j <= i; ++j) {
                    float s = A_sh[i * P + j];
                    if (i == j) {
                        s += diagonal_jitter;
                    }
                    #pragma unroll
                    for (int kk = 0; kk < P; ++kk) {
                        if (kk < j) s -= L_sh[i * P + kk] * L_sh[j * P + kk];
                    }
                    if (i == j) {
                        const float diagonal_scale = fabsf(A_sh[i * P + i] + diagonal_jitter);
                        const float pivot_floor = fmaxf(1.0e-12f, fminf(1.0e-5f, 1.0e-3f * diagonal_scale));
                        if (!(s > pivot_floor)) {
                            stable = 0;
                            s = pivot_floor;
                        }
                        factors[qrow * P * P + i * P + i] = s;
                        L_sh[i * P + i] = sqrtf(s);
                    } else {
                        const float denom = L_sh[j * P + j];
                        float val = 0.0f;
                        if (denom > 0.0f) {
                            val = s / denom;
                        } else {
                            stable = 0;
                        }
                        L_sh[i * P + j] = val;
                        factors[qrow * P * P + i * P + j] = val;
                    }
                }
                #pragma unroll
                for (int j = 0; j < P; ++j) {
                    if (j > i) {
                        L_sh[i * P + j] = 0.0f;
                        factors[qrow * P * P + i * P + j] = 0.0f;
                    }
                }
            }

            if (stable) {
                break;
            }
        }
        exact_counts[qrow] = exact_sh;
    }
    __syncthreads();

    const int64_t fb = fb_sh;
    if (fb < 0 || fb >= feature_batches) return;

    // ---- Exact-hit branch: average features of exact-hit neighbours ----
    if (exact_sh > 0) {
        for (int c = (int)threadIdx.x; c < channels; c += (int)blockDim.x) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if (squared_distances[qrow * K + j] <= exact_eps) {
                    sum += features[(fb * source_count + indices[qrow * K + j]) * channels + c];
                }
            }
            interpolated[qrow * channels + c] = sum / (float)exact_sh;
            #pragma unroll
            for (int d = 0; d < D; ++d)
                field_gradient[(qrow * D + d) * channels + c] = 0.0f;
        }
        return;
    }

    // ---- Phase 4: RHS + Cholesky solve (all threads, parallelised over C) ----
    // b[i] = sum_j wb[j,i] * f[j,c]  =  sum_j w_j * φ_j_i * f_j_c
    for (int c = (int)threadIdx.x; c < channels; c += (int)blockDim.x) {
        float b[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) b[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float f = features[(fb * source_count + indices[qrow * K + j]) * channels + c];
            #pragma unroll
            for (int i = 0; i < P; ++i) b[i] += wb_sh[j * P + i] * f;
        }
        // Forward substitution: L y = b
        float y[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = b[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) {
                if (j < i) s -= L_sh[i * P + j] * y[j];
            }
            y[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }
        // Back substitution: L^T x = y
        float x[P];
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = y[i];
            #pragma unroll
            for (int j = 0; j < P; ++j) {
                if (j > i) s -= L_sh[j * P + i] * x[j];
            }
            x[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }
        interpolated[qrow * channels + c] = x[0];
        #pragma unroll
        for (int d = 0; d < D; ++d)
            field_gradient[(qrow * D + d) * channels + c] = x[1 + d];
    }
}

template <int D, int K>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_mls_fused_forward(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    float regularization,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int64_t total_queries = displaced_points.size(0);
    const int64_t channels      = features.size(2);
    auto interpolated   = torch::empty({total_queries, channels},    features.options());
    auto field_gradient = torch::empty({total_queries, D, channels}, features.options());
    auto factors        = torch::empty({total_queries, P, P},        features.options());
    auto exact_counts   = torch::empty({total_queries},              indices.options().dtype(torch::kInt32));

    constexpr int threads = 128;
    mls_fused_forward_kernel<D, K><<<
        static_cast<unsigned int>(total_queries),
        threads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        displaced_points.data_ptr<float>(),
        neighbor_positions.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        squared_distances.data_ptr<float>(),
        features.data_ptr<float>(),
        feature_batch.data_ptr<int64_t>(),
        interpolated.data_ptr<float>(),
        field_gradient.data_ptr<float>(),
        factors.data_ptr<float>(),
        exact_counts.data_ptr<int32_t>(),
        static_cast<int>(total_queries),
        static_cast<int>(features.size(0)),
        static_cast<int>(features.size(1)),
        static_cast<int>(channels),
        regularization,
        bandwidth_min,
        exact_eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {interpolated, field_gradient, factors, exact_counts};
}

// Cooperative fused MLS backward kernel.
//
// Inputs:  saved forward state (displaced_points, neighbor_positions, indices,
//          squared_distances, features, feature_batch, factors, exact_counts)
//          and upstream gradients (d_interpolated, d_field_gradient or nullptr).
// Outputs: d_features (pre-zeroed by caller via torch::zeros_like) and
//          d_displaced_points (written directly; no pre-zero required).
//
// Per-block (one block per query row):
//   Phase 0 (thread 0):  load L from saved factors, compute h, zero d_q_sh.
//   Phase 1 (0..K-1):    recompute basis and wb (same as forward).
//   Exact-hit path:      scatter d_interpolated / exact_count to exact neighbors;
//                        write d_displaced = 0.
//   Non-exact path (all threads, channel-parallel):
//     For each channel c:
//       1. Solve A G[:,c] = d_x[:,c] using saved L (L y = d_x, L^T G = y).
//       2. Recompute forward coefficients x[:,c] from saved L and gathered features.
//       3. For each neighbor j: atomicAdd wb_j·G to d_features[src_j,c].
//       4. Accumulate per-channel contribution to partial_dq[d] in registers,
//          then atomicAdd to shared d_q_sh[d] at end of channel stripe.
//   Final sync: thread d writes d_q_sh[d] to d_displaced_points[qrow,d].

template <int D, int K>
__global__ void mls_fused_backward_kernel(
    const float*    __restrict__ displaced_points,   // (M, D)
    const float*    __restrict__ neighbor_positions, // (M, K, D)
    const int64_t*  __restrict__ indices,            // (M, K)
    const float*    __restrict__ squared_distances,  // (M, K)
    const float*    __restrict__ features,           // (B, N, C)
    const int64_t*  __restrict__ feature_batch,      // (M,)
    const float*    __restrict__ factors,            // (M, P, P) saved Cholesky L
    const int32_t*  __restrict__ exact_counts,       // (M,)
    const float*    __restrict__ d_interpolated,     // (M, C)
    const float*    __restrict__ d_field_gradient,   // (M, D, C) or nullptr
    float*          __restrict__ d_features,         // (B, N, C) — pre-zeroed
    float*          __restrict__ d_displaced_points, // (M, D)
    int total_queries,
    int feature_batches,
    int source_count,
    int channels,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int qrow = blockIdx.x;
    if (qrow >= total_queries) return;

    __shared__ float   bw_sh;
    __shared__ int64_t fb_sh;
    __shared__ int     exact_sh;
    __shared__ float   basis_sh[K * P];
    __shared__ float   wb_sh[K * P];
    __shared__ float   L_sh[P * P];
    __shared__ float   d_q_sh[D];

    __shared__ int ill_cond_sh;  // 1 if any L diagonal below threshold → skip MLS backward

    // ---- Phase 0: scalars and L recovery (thread 0) ----
    if (threadIdx.x == 0) {
        fb_sh    = (feature_batch != nullptr) ? feature_batch[qrow] : 0LL;
        bw_sh    = fmaxf(squared_distances[qrow * K + ((K - 1) / 2)], bandwidth_min);
        exact_sh = exact_counts[qrow];
        // Recover L from stored factors:
        //   diagonal: stored raw residual s → L[i,i] = sqrt(max(s,0))
        //   lower off-diagonal: stored directly
        //   upper off-diagonal: 0
        ill_cond_sh = 0;
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            #pragma unroll
            for (int j = 0; j < P; ++j) {
                const float stored = factors[qrow * P * P + i * P + j];
                if (j > i) {
                    L_sh[i * P + j] = 0.0f;
                } else if (i == j) {
                    // Guard against legacy or corrupted saved factors that would
                    // reconstruct a zero Cholesky diagonal.
                    if (stored <= 0.0f) ill_cond_sh = 1;
                    L_sh[i * P + i] = sqrtf(fmaxf(stored, 0.0f));
                } else {
                    L_sh[i * P + j] = stored;
                }
            }
        }
        #pragma unroll
        for (int d = 0; d < D; ++d) d_q_sh[d] = 0.0f;
    }
    __syncthreads();

    // ---- Phase 1: recompute basis and wb (threads 0..K-1) ----
    if ((int)threadIdx.x < K) {
        const int j = (int)threadIdx.x;
        float delta[D];
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const float dv = displaced_points[qrow * D + d]
                           - neighbor_positions[(qrow * K + j) * D + d];
            delta[d] = dv;
            dsq     += dv * dv;
        }
        const float w   = expf(-dsq / (2.0f * bw_sh));
        basis_sh[j * P] = 1.0f;
        wb_sh[j * P]    = w;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            basis_sh[j * P + 1 + d] = delta[d];
            wb_sh[j * P + 1 + d]    = w * delta[d];
        }
    }
    __syncthreads();

    const int64_t fb = fb_sh;

    if (fb < 0 || fb >= feature_batches) {
        for (int d = (int)threadIdx.x; d < D; d += (int)blockDim.x)
            d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    // ---- Exact-hit branch ----
    if (exact_sh > 0) {
        for (int c = (int)threadIdx.x; c < channels; c += (int)blockDim.x) {
            const float scale = d_interpolated[qrow * channels + c] / (float)exact_sh;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if (squared_distances[qrow * K + j] <= exact_eps) {
                    atomicAdd(
                        &d_features[(fb * source_count + indices[qrow * K + j]) * channels + c],
                        scale
                    );
                }
            }
        }
        for (int d = (int)threadIdx.x; d < D; d += (int)blockDim.x)
            d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    // ---- Ill-conditioned guard: zero gradients when Cholesky is near-singular ----
    if (ill_cond_sh) {
        for (int d = (int)threadIdx.x; d < D; d += (int)blockDim.x)
            d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    // ---- Non-exact path: channel-parallel implicit-diff backward ----
    for (int c = (int)threadIdx.x; c < channels; c += (int)blockDim.x) {

        // 1. Form d_x[P]: upstream gradient w.r.t. per-query MLS coefficients.
        float dx[P];
        dx[0] = d_interpolated[qrow * channels + c];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dx[1 + d] = (d_field_gradient != nullptr)
                       ? d_field_gradient[(qrow * D + d) * channels + c]
                       : 0.0f;
        }

        // 2. Solve A G[:,c] = d_x  (A symmetric, so A = A^T).
        //    Forward sub: L y = d_x.  Back sub: L^T G = y.
        float ytmp[P], G[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = dx[i];
            #pragma unroll
            for (int jj = 0; jj < P; ++jj) {
                if (jj < i) s -= L_sh[i * P + jj] * ytmp[jj];
            }
            ytmp[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = ytmp[i];
            #pragma unroll
            for (int jj = 0; jj < P; ++jj) {
                if (jj > i) s -= L_sh[jj * P + i] * G[jj];
            }
            G[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }

        // 3. Recompute forward coefficients x[:,c] using saved L.
        //    RHS: b[i] = Σ_j wb_ji * f_j[c].  Solve L yx = b, L^T x = yx.
        float b[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) b[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float fval =
                features[(fb * source_count + indices[qrow * K + j]) * channels + c];
            #pragma unroll
            for (int i = 0; i < P; ++i) b[i] += wb_sh[j * P + i] * fval;
        }
        float ytmp2[P], x[P];
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float s = b[i];
            #pragma unroll
            for (int jj = 0; jj < P; ++jj) {
                if (jj < i) s -= L_sh[i * P + jj] * ytmp2[jj];
            }
            ytmp2[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }
        #pragma unroll
        for (int ii = 0; ii < P; ++ii) {
            const int i = P - 1 - ii;
            float s = ytmp2[i];
            #pragma unroll
            for (int jj = 0; jj < P; ++jj) {
                if (jj > i) s -= L_sh[jj * P + i] * x[jj];
            }
            x[i] = s / fmaxf(L_sh[i * P + i], 1.0e-20f);
        }

        // 4. Per-neighbor: scatter d_features and accumulate partial d_displaced.
        float partial_dq[D];
        #pragma unroll
        for (int d = 0; d < D; ++d) partial_dq[d] = 0.0f;

        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const int64_t src = fb * source_count + indices[qrow * K + j];
            const float fval  = features[src * channels + c];
            const float w_j   = wb_sh[j * P];   // wb[j,0] = w_j * φ_j[0] = w_j

            // d_features[src, c] += Σ_i wb_ji * G_i
            float df = 0.0f;
            #pragma unroll
            for (int i = 0; i < P; ++i) df += wb_sh[j * P + i] * G[i];
            atomicAdd(&d_features[src * channels + c], df);

            // Quantities for d_displaced:
            //   ay_jc     = Σ_i φ_ji * G_i
            //   fitted_jc = Σ_l φ_jl * x_l
            //   residual  = f_j[c] - fitted_jc
            float ay_jc = 0.0f, fitted_jc = 0.0f;
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                ay_jc    += basis_sh[j * P + i] * G[i];
                fitted_jc += basis_sh[j * P + i] * x[i];
            }
            const float residual_jc = fval - fitted_jc;
            const float ay_res      = ay_jc * residual_jc;

            // grad_delta_j[d] = w_j*(residual_jc*G[d+1] - ay_jc*x[d+1])
            //                   - ay_res*w_j*δ_jd / h
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                const float delta_jd = basis_sh[j * P + 1 + d];
                partial_dq[d] += w_j * (residual_jc * G[1 + d] - ay_jc * x[1 + d])
                               - ay_res * w_j * delta_jd / bw_sh;
            }
        }

        // Reduce partial_dq into shared accumulator across channel stripes.
        #pragma unroll
        for (int d = 0; d < D; ++d) atomicAdd(&d_q_sh[d], partial_dq[d]);
    }
    __syncthreads();

    // Write d_displaced_points from shared accumulator.
    if ((int)threadIdx.x < D) {
        d_displaced_points[qrow * D + (int)threadIdx.x] = d_q_sh[(int)threadIdx.x];
    }
}

template <int D, int K>
static std::tuple<torch::Tensor, torch::Tensor>
launch_mls_fused_backward(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,  // numel()==0 signals "not used"
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int64_t total_queries    = displaced_points.size(0);
    const int64_t channels         = features.size(2);
    const int64_t feature_batches  = features.size(0);
    const int64_t source_count     = features.size(1);

    auto d_features       = torch::zeros_like(features);
    auto d_displaced      = torch::empty_like(displaced_points);

    const float* d_field_grad_ptr =
        (d_field_gradient.numel() > 0) ? d_field_gradient.data_ptr<float>() : nullptr;

    constexpr int threads = 128;
    mls_fused_backward_kernel<D, K><<<
        static_cast<unsigned int>(total_queries),
        threads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        displaced_points.data_ptr<float>(),
        neighbor_positions.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        squared_distances.data_ptr<float>(),
        features.data_ptr<float>(),
        feature_batch.data_ptr<int64_t>(),
        factors.data_ptr<float>(),
        exact_counts.data_ptr<int32_t>(),
        d_interpolated.data_ptr<float>(),
        d_field_grad_ptr,
        d_features.data_ptr<float>(),
        d_displaced.data_ptr<float>(),
        static_cast<int>(total_queries),
        static_cast<int>(feature_batches),
        static_cast<int>(source_count),
        static_cast<int>(channels),
        bandwidth_min,
        exact_eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_features, d_displaced};
}

std::tuple<torch::Tensor, torch::Tensor> mls_fused_backward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    double bandwidth_min,
    double exact_eps
) {
    TORCH_CHECK(displaced_points.is_cuda(),         "mls_fused_backward: displaced_points must be CUDA");
    TORCH_CHECK(neighbor_positions.is_cuda(),       "mls_fused_backward: neighbor_positions must be CUDA");
    TORCH_CHECK(indices.is_cuda(),                  "mls_fused_backward: indices must be CUDA");
    TORCH_CHECK(squared_distances.is_cuda(),        "mls_fused_backward: squared_distances must be CUDA");
    TORCH_CHECK(features.is_cuda(),                 "mls_fused_backward: features must be CUDA");
    TORCH_CHECK(feature_batch.is_cuda(),            "mls_fused_backward: feature_batch must be CUDA");
    TORCH_CHECK(factors.is_cuda(),                  "mls_fused_backward: factors must be CUDA");
    TORCH_CHECK(exact_counts.is_cuda(),             "mls_fused_backward: exact_counts must be CUDA");
    TORCH_CHECK(d_interpolated.is_cuda(),           "mls_fused_backward: d_interpolated must be CUDA");
    TORCH_CHECK(displaced_points.is_contiguous(),   "mls_fused_backward: displaced_points must be contiguous");
    TORCH_CHECK(neighbor_positions.is_contiguous(), "mls_fused_backward: neighbor_positions must be contiguous");
    TORCH_CHECK(indices.is_contiguous(),            "mls_fused_backward: indices must be contiguous");
    TORCH_CHECK(squared_distances.is_contiguous(),  "mls_fused_backward: squared_distances must be contiguous");
    TORCH_CHECK(features.is_contiguous(),           "mls_fused_backward: features must be contiguous");
    TORCH_CHECK(feature_batch.is_contiguous(),      "mls_fused_backward: feature_batch must be contiguous");
    TORCH_CHECK(factors.is_contiguous(),            "mls_fused_backward: factors must be contiguous");
    TORCH_CHECK(exact_counts.is_contiguous(),       "mls_fused_backward: exact_counts must be contiguous");
    TORCH_CHECK(d_interpolated.is_contiguous(),     "mls_fused_backward: d_interpolated must be contiguous");
    TORCH_CHECK(displaced_points.scalar_type()   == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(neighbor_positions.scalar_type() == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(squared_distances.scalar_type()  == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(features.scalar_type()           == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(factors.scalar_type()            == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(d_interpolated.scalar_type()     == torch::kFloat32, "mls_fused_backward: float32 required");
    TORCH_CHECK(indices.scalar_type()    == torch::kInt64,  "mls_fused_backward: indices must be int64");
    TORCH_CHECK(feature_batch.scalar_type() == torch::kInt64, "mls_fused_backward: feature_batch must be int64");
    TORCH_CHECK(exact_counts.scalar_type() == torch::kInt32, "mls_fused_backward: exact_counts must be int32");
    TORCH_CHECK(displaced_points.size(1) == 2 || displaced_points.size(1) == 3,
        "mls_fused_backward: D must be 2 or 3");
    TORCH_CHECK(
        neighbor_positions.size(1) == 4 || neighbor_positions.size(1) == 8 || neighbor_positions.size(1) == 16,
        "mls_fused_backward: K must be 4, 8, or 16");
    if (d_field_gradient.numel() > 0) {
        TORCH_CHECK(d_field_gradient.is_cuda(),       "mls_fused_backward: d_field_gradient must be CUDA");
        TORCH_CHECK(d_field_gradient.is_contiguous(), "mls_fused_backward: d_field_gradient must be contiguous");
        TORCH_CHECK(d_field_gradient.scalar_type() == torch::kFloat32,
            "mls_fused_backward: d_field_gradient must be float32");
    }

    c10::cuda::CUDAGuard device_guard(features.device());

    const int dim = static_cast<int>(displaced_points.size(1));
    const int k   = static_cast<int>(neighbor_positions.size(1));
    if (dim == 2 && k ==  4) return launch_mls_fused_backward<2,  4>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
    if (dim == 2 && k ==  8) return launch_mls_fused_backward<2,  8>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
    if (dim == 2 && k == 16) return launch_mls_fused_backward<2, 16>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
    if (dim == 3 && k ==  4) return launch_mls_fused_backward<3,  4>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
    if (dim == 3 && k ==  8) return launch_mls_fused_backward<3,  8>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
    return                        launch_mls_fused_backward<3, 16>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, factors, exact_counts, d_interpolated, d_field_gradient, (float)bandwidth_min, (float)exact_eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mls_fused_forward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    double regularization,
    double bandwidth_min,
    double exact_eps
) {
    TORCH_CHECK(displaced_points.is_cuda(),         "mls_fused_forward: displaced_points must be CUDA");
    TORCH_CHECK(neighbor_positions.is_cuda(),       "mls_fused_forward: neighbor_positions must be CUDA");
    TORCH_CHECK(indices.is_cuda(),                  "mls_fused_forward: indices must be CUDA");
    TORCH_CHECK(squared_distances.is_cuda(),        "mls_fused_forward: squared_distances must be CUDA");
    TORCH_CHECK(features.is_cuda(),                 "mls_fused_forward: features must be CUDA");
    TORCH_CHECK(feature_batch.is_cuda(),            "mls_fused_forward: feature_batch must be CUDA");
    TORCH_CHECK(displaced_points.is_contiguous(),   "mls_fused_forward: displaced_points must be contiguous");
    TORCH_CHECK(neighbor_positions.is_contiguous(), "mls_fused_forward: neighbor_positions must be contiguous");
    TORCH_CHECK(indices.is_contiguous(),            "mls_fused_forward: indices must be contiguous");
    TORCH_CHECK(squared_distances.is_contiguous(),  "mls_fused_forward: squared_distances must be contiguous");
    TORCH_CHECK(features.is_contiguous(),           "mls_fused_forward: features must be contiguous");
    TORCH_CHECK(feature_batch.is_contiguous(),      "mls_fused_forward: feature_batch must be contiguous");
    TORCH_CHECK(displaced_points.scalar_type()   == torch::kFloat32, "mls_fused_forward: displaced_points must be float32");
    TORCH_CHECK(neighbor_positions.scalar_type() == torch::kFloat32, "mls_fused_forward: neighbor_positions must be float32");
    TORCH_CHECK(squared_distances.scalar_type()  == torch::kFloat32, "mls_fused_forward: squared_distances must be float32");
    TORCH_CHECK(features.scalar_type()           == torch::kFloat32, "mls_fused_forward: features must be float32");
    TORCH_CHECK(indices.scalar_type()            == torch::kInt64,   "mls_fused_forward: indices must be int64");
    TORCH_CHECK(feature_batch.scalar_type()      == torch::kInt64,   "mls_fused_forward: feature_batch must be int64");
    TORCH_CHECK(displaced_points.dim()   == 2, "mls_fused_forward: displaced_points must have shape (M, D)");
    TORCH_CHECK(neighbor_positions.dim() == 3, "mls_fused_forward: neighbor_positions must have shape (M, K, D)");
    TORCH_CHECK(indices.dim()            == 2, "mls_fused_forward: indices must have shape (M, K)");
    TORCH_CHECK(squared_distances.dim()  == 2, "mls_fused_forward: squared_distances must have shape (M, K)");
    TORCH_CHECK(features.dim()           == 3, "mls_fused_forward: features must have shape (B, N, C)");
    TORCH_CHECK(feature_batch.dim()      == 1, "mls_fused_forward: feature_batch must have shape (M,)");
    TORCH_CHECK(displaced_points.size(0) == neighbor_positions.size(0), "mls_fused_forward: M mismatch");
    TORCH_CHECK(displaced_points.size(0) == indices.size(0),            "mls_fused_forward: M mismatch");
    TORCH_CHECK(displaced_points.size(0) == squared_distances.size(0),  "mls_fused_forward: M mismatch");
    TORCH_CHECK(displaced_points.size(0) == feature_batch.size(0),      "mls_fused_forward: M mismatch");
    TORCH_CHECK(displaced_points.size(1) == neighbor_positions.size(2), "mls_fused_forward: D mismatch");
    TORCH_CHECK(indices.size(1)          == neighbor_positions.size(1), "mls_fused_forward: K mismatch");
    TORCH_CHECK(squared_distances.size(1)== neighbor_positions.size(1), "mls_fused_forward: K mismatch");
    TORCH_CHECK(displaced_points.size(1) == 2 || displaced_points.size(1) == 3,
        "mls_fused_forward: D must be 2 or 3");
    TORCH_CHECK(
        neighbor_positions.size(1) == 4 || neighbor_positions.size(1) == 8 || neighbor_positions.size(1) == 16,
        "mls_fused_forward: K must be 4, 8, or 16");
    TORCH_CHECK(features.size(2) >= 1, "mls_fused_forward: features must have at least one channel");
    TORCH_CHECK(displaced_points.device()  == features.device(), "mls_fused_forward: device mismatch");
    TORCH_CHECK(neighbor_positions.device()== features.device(), "mls_fused_forward: device mismatch");
    TORCH_CHECK(indices.device()           == features.device(), "mls_fused_forward: device mismatch");
    TORCH_CHECK(squared_distances.device() == features.device(), "mls_fused_forward: device mismatch");
    TORCH_CHECK(feature_batch.device()     == features.device(), "mls_fused_forward: device mismatch");

    c10::cuda::CUDAGuard device_guard(features.device());

    const int dim = static_cast<int>(displaced_points.size(1));
    const int k   = static_cast<int>(neighbor_positions.size(1));
    if (dim == 2 && k ==  4) return launch_mls_fused_forward<2,  4>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
    if (dim == 2 && k ==  8) return launch_mls_fused_forward<2,  8>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
    if (dim == 2 && k == 16) return launch_mls_fused_forward<2, 16>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
    if (dim == 3 && k ==  4) return launch_mls_fused_forward<3,  4>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
    if (dim == 3 && k ==  8) return launch_mls_fused_forward<3,  8>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
    return                        launch_mls_fused_forward<3, 16>(displaced_points, neighbor_positions, indices, squared_distances, features, feature_batch, (float)regularization, (float)bandwidth_min, (float)exact_eps);
}
