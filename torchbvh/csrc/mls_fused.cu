#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cmath>
#include <tuple>

// Production indexed MLS kernels: packed for narrow features and cooperative
// for wide features. Only Indexed=true specializations are instantiated.

template <int D>
__device__ __forceinline__ float packed_basis_value(
    const float* __restrict__ delta,
    int j,
    int i
) {
    return i == 0 ? 1.0f : delta[j * D + i - 1];
}

template <int D, int K, bool Indexed = false>
__global__ void mls_packed_forward_kernel(
    const float* __restrict__ displaced_points,
    const float* __restrict__ neighbor_positions,
    const int64_t* __restrict__ indices,
    const float* __restrict__ squared_distances,
    const float* __restrict__ features,
    const int64_t* __restrict__ feature_batch,
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
    float exact_eps
) {
    constexpr int P = D + 1;
    const int storage_qrow = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (storage_qrow >= total_queries) return;
    const int sample = Indexed ? storage_qrow / queries_per_batch : 0;
    const int qrow = Indexed
        ? sample * queries_per_batch + (int)query_order[storage_qrow]
        : storage_qrow;
    const int64_t fb = Indexed
        ? (int64_t)sample * (queries_per_batch / queries_per_head)
            + ((qrow - sample * queries_per_batch) / queries_per_head)
        : feature_batch[qrow];
    if (fb < 0 || fb >= feature_batches) return;
    const float bandwidth = fmaxf(
        squared_distances[storage_qrow * K + ((K - 1) / 2)], bandwidth_min);

    float delta[K * D];
    float weights[K];
    int exact_count = 0;
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = Indexed
                ? source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d]
                : neighbor_positions[(qrow * K + j) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
            delta[j * D + d] = dv;
            dsq += dv * dv;
        }
        weights[j] = expf(-dsq / (2.0f * bandwidth));
        exact_count += squared_distances[storage_qrow * K + j] <= exact_eps;
    }
    exact_counts[storage_qrow] = exact_count;

    if (exact_count > 0) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if (squared_distances[storage_qrow * K + j] <= exact_eps) {
                    sum += features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
                }
            }
            interpolated[qrow * channels + c] = sum / (float)exact_count;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                field_gradient[(qrow * D + d) * channels + c] = 0.0f;
            }
        }
        return;
    }

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
    float L[P * P];
    #pragma unroll
    for (int attempt = 0; attempt < 2; ++attempt) {
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
                    factors[storage_qrow * P * P + i * P + i] = s;
                    L[i * P + i] = sqrtf(s);
                } else {
                    const float denom = L[j * P + j];
                    const float value = denom > 0.0f ? s / denom : 0.0f;
                    stable &= denom > 0.0f;
                    L[i * P + j] = value;
                    factors[storage_qrow * P * P + i * P + j] = value;
                }
            }
            #pragma unroll
            for (int j = 0; j < P; ++j) {
                if (j > i) factors[storage_qrow * P * P + i * P + j] = 0.0f;
            }
        }
        if (stable) break;
    }

    for (int c = 0; c < channels; ++c) {
        float rhs[P] = {};
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float f = features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
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
        interpolated[qrow * channels + c] = x[0];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            field_gradient[(qrow * D + d) * channels + c] = x[d + 1];
        }
    }
}

template <int D, int K, bool Indexed = false>
__global__ void mls_packed_backward_kernel(
    const float* __restrict__ displaced_points,
    const float* __restrict__ neighbor_positions,
    const int64_t* __restrict__ indices,
    const float* __restrict__ squared_distances,
    const float* __restrict__ features,
    const int64_t* __restrict__ feature_batch,
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
    float exact_eps
) {
    constexpr int P = D + 1;
    const int storage_qrow = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (storage_qrow >= total_queries) return;
    const int sample = Indexed ? storage_qrow / queries_per_batch : 0;
    const int qrow = Indexed
        ? sample * queries_per_batch + (int)query_order[storage_qrow]
        : storage_qrow;
    const int64_t fb = Indexed
        ? (int64_t)sample * (queries_per_batch / queries_per_head)
            + ((qrow - sample * queries_per_batch) / queries_per_head)
        : feature_batch[qrow];
    if (fb < 0 || fb >= feature_batches) {
        #pragma unroll
        for (int d = 0; d < D; ++d) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }
    const int exact_count = exact_counts[storage_qrow];
    if (exact_count > 0) {
        for (int c = 0; c < channels; ++c) {
            const float scale = d_interpolated[qrow * channels + c] / (float)exact_count;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                if (squared_distances[storage_qrow * K + j] <= exact_eps) {
                    atomicAdd(&d_features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c], scale);
                }
            }
        }
        #pragma unroll
        for (int d = 0; d < D; ++d) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    float L[P * P];
    int ill_conditioned = 0;
    #pragma unroll
    for (int i = 0; i < P; ++i) {
        #pragma unroll
        for (int j = 0; j < P; ++j) {
            const float stored = factors[storage_qrow * P * P + i * P + j];
            if (j > i) L[i * P + j] = 0.0f;
            else if (i == j) {
                ill_conditioned |= stored <= 0.0f;
                L[i * P + i] = sqrtf(fmaxf(stored, 0.0f));
            } else L[i * P + j] = stored;
        }
    }
    if (ill_conditioned) {
        #pragma unroll
        for (int d = 0; d < D; ++d) d_displaced_points[qrow * D + d] = 0.0f;
        return;
    }

    const float bandwidth = fmaxf(
        squared_distances[storage_qrow * K + ((K - 1) / 2)], bandwidth_min);
    float delta[K * D];
    float weights[K];
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        float dsq = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = Indexed
                ? source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d]
                : neighbor_positions[(qrow * K + j) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
            delta[j * D + d] = dv;
            dsq += dv * dv;
        }
        weights[j] = expf(-dsq / (2.0f * bandwidth));
    }

    float dq[D] = {};
    for (int c = 0; c < channels; ++c) {
        float dx[P];
        dx[0] = d_interpolated[qrow * channels + c];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dx[d + 1] = d_field_gradient == nullptr
                ? 0.0f : d_field_gradient[(qrow * D + d) * channels + c];
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

        float rhs[P] = {};
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float f = features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
            #pragma unroll
            for (int i = 0; i < P; ++i) rhs[i] += weights[j] * packed_basis_value<D>(delta, j, i) * f;
        }
        float yx[P], x[P];
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

        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const int64_t src = fb * source_count + indices[storage_qrow * K + j];
            const float f = features[src * channels + c];
            float df = 0.0f, ay = 0.0f, fitted = 0.0f;
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                const float phi = packed_basis_value<D>(delta, j, i);
                df += weights[j] * phi * G[i];
                ay += phi * G[i];
                fitted += phi * x[i];
            }
            atomicAdd(&d_features[src * channels + c], df);
            const float residual = f - fitted;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dq[d] += weights[j] * (residual * G[d + 1] - ay * x[d + 1])
                    - ay * residual * weights[j] * delta[j * D + d] / bandwidth;
            }
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) d_displaced_points[qrow * D + d] = dq[d];
}

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

template <int D, int K, bool Indexed = false>
__global__ void mls_fused_forward_kernel(
    const float* __restrict__ displaced_points,   // (M, D)
    const float* __restrict__ neighbor_positions, // (M, K, D)
    const int64_t* __restrict__ indices,          // (M, K)
    const float* __restrict__ squared_distances,  // (M, K)
    const float* __restrict__ features,           // (B, N, C)
    const int64_t* __restrict__ feature_batch,    // (M,)
    const float* __restrict__ source_points,
    const int64_t* __restrict__ query_order,
    float* __restrict__ interpolated,             // (M, C)
    float* __restrict__ field_gradient,           // (M, D, C)
    float* __restrict__ factors,                  // (M, P, P) — Cholesky L, lower-triangular
    int32_t* __restrict__ exact_counts,           // (M,)
    int total_queries,
    int feature_batches,
    int source_count,
    int channels,
    int queries_per_batch,
    int queries_per_head,
    float regularization,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int storage_qrow = blockIdx.x;
    if (storage_qrow >= total_queries) return;
    const int sample = Indexed ? storage_qrow / queries_per_batch : 0;
    const int qrow = Indexed
        ? sample * queries_per_batch + (int)query_order[storage_qrow]
        : storage_qrow;

    __shared__ float   bw_sh;
    __shared__ int64_t fb_sh;
    __shared__ int     exact_sh;
    __shared__ float   basis_sh[K * P];   // φ_j_i  (unweighted)
    __shared__ float   wb_sh[K * P];      // w_j * φ_j_i  (matches reference numerics)
    __shared__ float   A_sh[P * P];       // normal matrix (written by P threads)
    __shared__ float   L_sh[P * P];       // Cholesky factor (written by thread 0)

    // ---- Phase 0: scalars (thread 0 only) ----
    if (threadIdx.x == 0) {
        fb_sh = Indexed
            ? (int64_t)sample * (queries_per_batch / queries_per_head)
                + ((qrow - sample * queries_per_batch) / queries_per_head)
            : ((feature_batch != nullptr) ? feature_batch[qrow] : 0LL);
        bw_sh = fmaxf(
            squared_distances[storage_qrow * K + ((K - 1) / 2)], bandwidth_min);
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
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = Indexed
                ? source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d]
                : neighbor_positions[(qrow * K + j) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
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
        if (squared_distances[storage_qrow * K + j] <= exact_eps) atomicAdd(&exact_sh, 1);
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
                        factors[storage_qrow * P * P + i * P + i] = s;
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
                        factors[storage_qrow * P * P + i * P + j] = val;
                    }
                }
                #pragma unroll
                for (int j = 0; j < P; ++j) {
                    if (j > i) {
                        L_sh[i * P + j] = 0.0f;
                        factors[storage_qrow * P * P + i * P + j] = 0.0f;
                    }
                }
            }

            if (stable) {
                break;
            }
        }
        exact_counts[storage_qrow] = exact_sh;
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
                if (squared_distances[storage_qrow * K + j] <= exact_eps) {
                    sum += features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
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
            const float f = features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
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

template <int D, int K, bool Indexed = false>
__global__ void mls_fused_backward_kernel(
    const float*    __restrict__ displaced_points,   // (M, D)
    const float*    __restrict__ neighbor_positions, // (M, K, D)
    const int64_t*  __restrict__ indices,            // (M, K)
    const float*    __restrict__ squared_distances,  // (M, K)
    const float*    __restrict__ features,           // (B, N, C)
    const int64_t*  __restrict__ feature_batch,      // (M,)
    const float*    __restrict__ source_points,
    const int64_t*  __restrict__ query_order,
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
    int queries_per_batch,
    int queries_per_head,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int storage_qrow = blockIdx.x;
    if (storage_qrow >= total_queries) return;
    const int sample = Indexed ? storage_qrow / queries_per_batch : 0;
    const int qrow = Indexed
        ? sample * queries_per_batch + (int)query_order[storage_qrow]
        : storage_qrow;

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
        fb_sh = Indexed
            ? (int64_t)sample * (queries_per_batch / queries_per_head)
                + ((qrow - sample * queries_per_batch) / queries_per_head)
            : ((feature_batch != nullptr) ? feature_batch[qrow] : 0LL);
        bw_sh = fmaxf(
            squared_distances[storage_qrow * K + ((K - 1) / 2)], bandwidth_min);
        exact_sh = exact_counts[storage_qrow];
        // Recover L from stored factors:
        //   diagonal: stored raw residual s → L[i,i] = sqrt(max(s,0))
        //   lower off-diagonal: stored directly
        //   upper off-diagonal: 0
        ill_cond_sh = 0;
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            #pragma unroll
            for (int j = 0; j < P; ++j) {
                const float stored = factors[storage_qrow * P * P + i * P + j];
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
            const int64_t source_idx = indices[storage_qrow * K + j];
            const float neighbor = Indexed
                ? source_points[(static_cast<int64_t>(sample) * source_count + source_idx) * D + d]
                : neighbor_positions[(qrow * K + j) * D + d];
            const float dv = displaced_points[qrow * D + d] - neighbor;
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
                if (squared_distances[storage_qrow * K + j] <= exact_eps) {
                    atomicAdd(
                        &d_features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c],
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
                features[(fb * source_count + indices[storage_qrow * K + j]) * channels + c];
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
            const int64_t src = fb * source_count + indices[storage_qrow * K + j];
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
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_mls_packed_indexed_forward(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    int queries_per_batch,
    int queries_per_head,
    float regularization,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int total_queries = static_cast<int>(displaced_points.size(0));
    const int channels = static_cast<int>(features.size(2));
    auto interpolated = torch::empty({total_queries, channels}, features.options());
    auto field_gradient = torch::empty({total_queries, D, channels}, features.options());
    auto factors = torch::empty({total_queries, P, P}, features.options());
    auto exact_counts = torch::empty({total_queries}, indices.options().dtype(torch::kInt32));
    constexpr int threads = 128;
    const int blocks = (total_queries + threads - 1) / threads;
    mls_packed_forward_kernel<D, K, true><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        displaced_points.data_ptr<float>(), nullptr,
        indices.data_ptr<int64_t>(), squared_distances.data_ptr<float>(),
        features.data_ptr<float>(), nullptr,
        source_points.data_ptr<float>(), query_order.data_ptr<int64_t>(),
        interpolated.data_ptr<float>(), field_gradient.data_ptr<float>(),
        factors.data_ptr<float>(), exact_counts.data_ptr<int32_t>(),
        total_queries, static_cast<int>(features.size(0)),
        static_cast<int>(features.size(1)), channels,
        queries_per_batch, queries_per_head,
        regularization, bandwidth_min, exact_eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {interpolated, field_gradient, factors, exact_counts};
}

template <int D, int K>
static std::tuple<torch::Tensor, torch::Tensor>
launch_mls_packed_indexed_backward(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    int queries_per_batch,
    int queries_per_head,
    float bandwidth_min,
    float exact_eps
) {
    const int total_queries = static_cast<int>(displaced_points.size(0));
    const int channels = static_cast<int>(features.size(2));
    auto d_features = torch::zeros_like(features);
    auto d_displaced = torch::empty_like(displaced_points);
    const float* d_field_ptr = d_field_gradient.numel() > 0
        ? d_field_gradient.data_ptr<float>() : nullptr;
    constexpr int threads = 128;
    const int blocks = (total_queries + threads - 1) / threads;
    mls_packed_backward_kernel<D, K, true><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        displaced_points.data_ptr<float>(), nullptr,
        indices.data_ptr<int64_t>(), squared_distances.data_ptr<float>(),
        features.data_ptr<float>(), nullptr,
        source_points.data_ptr<float>(), query_order.data_ptr<int64_t>(),
        factors.data_ptr<float>(), exact_counts.data_ptr<int32_t>(),
        d_interpolated.data_ptr<float>(), d_field_ptr,
        d_features.data_ptr<float>(), d_displaced.data_ptr<float>(),
        total_queries, static_cast<int>(features.size(0)),
        static_cast<int>(features.size(1)), channels,
        queries_per_batch, queries_per_head, bandwidth_min, exact_eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_features, d_displaced};
}

// Wide-channel indexed specialization. It retains Morton-ordered storage and
// in-kernel source-position loads, while assigning one cooperative block to a
// query so feature channels are processed in parallel.
template <int D, int K>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_mls_cooperative_indexed_forward(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    int queries_per_batch,
    int queries_per_head,
    float regularization,
    float bandwidth_min,
    float exact_eps
) {
    constexpr int P = D + 1;
    const int total_queries = static_cast<int>(displaced_points.size(0));
    const int channels = static_cast<int>(features.size(2));
    auto interpolated = torch::empty({total_queries, channels}, features.options());
    auto field_gradient = torch::empty({total_queries, D, channels}, features.options());
    auto factors = torch::empty({total_queries, P, P}, features.options());
    auto exact_counts = torch::empty(
        {total_queries}, indices.options().dtype(torch::kInt32));
    constexpr int threads = 128;
    mls_fused_forward_kernel<D, K, true><<<
        static_cast<unsigned int>(total_queries), threads, 0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        displaced_points.data_ptr<float>(), nullptr,
        indices.data_ptr<int64_t>(), squared_distances.data_ptr<float>(),
        features.data_ptr<float>(), nullptr,
        source_points.data_ptr<float>(), query_order.data_ptr<int64_t>(),
        interpolated.data_ptr<float>(), field_gradient.data_ptr<float>(),
        factors.data_ptr<float>(), exact_counts.data_ptr<int32_t>(),
        total_queries, static_cast<int>(features.size(0)),
        static_cast<int>(features.size(1)), channels,
        queries_per_batch, queries_per_head,
        regularization, bandwidth_min, exact_eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {interpolated, field_gradient, factors, exact_counts};
}

template <int D, int K>
static std::tuple<torch::Tensor, torch::Tensor>
launch_mls_cooperative_indexed_backward(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    int queries_per_batch,
    int queries_per_head,
    float bandwidth_min,
    float exact_eps
) {
    const int total_queries = static_cast<int>(displaced_points.size(0));
    const int channels = static_cast<int>(features.size(2));
    auto d_features = torch::zeros_like(features);
    auto d_displaced = torch::empty_like(displaced_points);
    const float* d_field_ptr = d_field_gradient.numel() > 0
        ? d_field_gradient.data_ptr<float>() : nullptr;
    constexpr int threads = 128;
    mls_fused_backward_kernel<D, K, true><<<
        static_cast<unsigned int>(total_queries), threads, 0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        displaced_points.data_ptr<float>(), nullptr,
        indices.data_ptr<int64_t>(), squared_distances.data_ptr<float>(),
        features.data_ptr<float>(), nullptr,
        source_points.data_ptr<float>(), query_order.data_ptr<int64_t>(),
        factors.data_ptr<float>(), exact_counts.data_ptr<int32_t>(),
        d_interpolated.data_ptr<float>(), d_field_ptr,
        d_features.data_ptr<float>(), d_displaced.data_ptr<float>(),
        total_queries, static_cast<int>(features.size(0)),
        static_cast<int>(features.size(1)), channels,
        queries_per_batch, queries_per_head, bandwidth_min, exact_eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_features, d_displaced};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mls_packed_indexed_forward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    int queries_per_batch,
    int queries_per_head,
    double regularization,
    double bandwidth_min,
    double exact_eps
) {
    const char* op = "mls_packed_indexed_forward";
    TORCH_CHECK(displaced_points.is_cuda() && source_points.is_cuda()
                && indices.is_cuda() && squared_distances.is_cuda()
                && features.is_cuda() && query_order.is_cuda(), op, ": CUDA inputs required");
    TORCH_CHECK(displaced_points.is_contiguous() && source_points.is_contiguous()
                && indices.is_contiguous() && squared_distances.is_contiguous()
                && features.is_contiguous() && query_order.is_contiguous(),
                op, ": contiguous inputs required");
    TORCH_CHECK(features.size(2) >= 1,
                op, ": channels must be positive");
    TORCH_CHECK(queries_per_batch >= 1 && queries_per_head >= 1
                && queries_per_batch % queries_per_head == 0,
                op, ": invalid query layout");
    c10::cuda::CUDAGuard device_guard(features.device());
    const int dim = static_cast<int>(displaced_points.size(1));
    const int k = static_cast<int>(indices.size(1));
#define DISPATCH_INDEXED_FWD(D, K) \
    do { \
        if (features.size(2) <= 32) \
            return launch_mls_packed_indexed_forward<D, K>(displaced_points, source_points, indices, squared_distances, features, query_order, queries_per_batch, queries_per_head, (float)regularization, (float)bandwidth_min, (float)exact_eps); \
        return launch_mls_cooperative_indexed_forward<D, K>(displaced_points, source_points, indices, squared_distances, features, query_order, queries_per_batch, queries_per_head, (float)regularization, (float)bandwidth_min, (float)exact_eps); \
    } while (false)
    if (dim == 2 && k == 4) DISPATCH_INDEXED_FWD(2, 4);
    if (dim == 2 && k == 8) DISPATCH_INDEXED_FWD(2, 8);
    if (dim == 2 && k == 16) DISPATCH_INDEXED_FWD(2, 16);
    if (dim == 3 && k == 4) DISPATCH_INDEXED_FWD(3, 4);
    if (dim == 3 && k == 8) DISPATCH_INDEXED_FWD(3, 8);
    TORCH_CHECK(dim == 3 && k == 16, op, ": D must be 2/3 and K must be 4/8/16");
    DISPATCH_INDEXED_FWD(3, 16);
#undef DISPATCH_INDEXED_FWD
}

std::tuple<torch::Tensor, torch::Tensor> mls_packed_indexed_backward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor source_points,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor query_order,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    int queries_per_batch,
    int queries_per_head,
    double bandwidth_min,
    double exact_eps
) {
    const char* op = "mls_packed_indexed_backward";
    TORCH_CHECK(features.is_cuda() && displaced_points.is_cuda(), op, ": CUDA inputs required");
    TORCH_CHECK(features.size(2) >= 1,
                op, ": channels must be positive");
    c10::cuda::CUDAGuard device_guard(features.device());
    const int dim = static_cast<int>(displaced_points.size(1));
    const int k = static_cast<int>(indices.size(1));
#define DISPATCH_INDEXED_BWD(D, K) \
    do { \
        if (features.size(2) <= 32) \
            return launch_mls_packed_indexed_backward<D, K>(displaced_points, source_points, indices, squared_distances, features, query_order, factors, exact_counts, d_interpolated, d_field_gradient, queries_per_batch, queries_per_head, (float)bandwidth_min, (float)exact_eps); \
        return launch_mls_cooperative_indexed_backward<D, K>(displaced_points, source_points, indices, squared_distances, features, query_order, factors, exact_counts, d_interpolated, d_field_gradient, queries_per_batch, queries_per_head, (float)bandwidth_min, (float)exact_eps); \
    } while (false)
    if (dim == 2 && k == 4) DISPATCH_INDEXED_BWD(2, 4);
    if (dim == 2 && k == 8) DISPATCH_INDEXED_BWD(2, 8);
    if (dim == 2 && k == 16) DISPATCH_INDEXED_BWD(2, 16);
    if (dim == 3 && k == 4) DISPATCH_INDEXED_BWD(3, 4);
    if (dim == 3 && k == 8) DISPATCH_INDEXED_BWD(3, 8);
    TORCH_CHECK(dim == 3 && k == 16, op, ": D must be 2/3 and K must be 4/8/16");
    DISPATCH_INDEXED_BWD(3, 16);
#undef DISPATCH_INDEXED_BWD
}
