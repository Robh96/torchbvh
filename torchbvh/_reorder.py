"""Query spatial reordering utilities.

Provides Morton-code sort for batched query tensors so that spatially
close queries are assigned to adjacent thread lanes in the k-NN kernel,
reducing SIMD warp divergence.
"""

import torch
from torchbvh import _C


def _spread_bits_3d(x: torch.Tensor) -> torch.Tensor:
    """Spread a 10-bit integer's bits to positions 0, 3, 6, ..., 27.

    Produces a 30-bit value where each original bit occupies every 3rd slot,
    leaving two zero bits between each pair of original bits so that three
    such values can be OR-combined (at shifts 0, 1, 2) to form a 3D Morton code.
    """
    x = x & 0x000003FF
    x = (x | (x << 16)) & 0xFF0000FF
    x = (x | (x << 8))  & 0x0F00F00F
    x = (x | (x << 4))  & 0xC30C30C3
    x = (x | (x << 2))  & 0x49249249
    return x


def _spread_bits_2d(x: torch.Tensor) -> torch.Tensor:
    """Spread a 16-bit integer's bits to even bit positions 0, 2, 4, ..., 30."""
    x = x & 0x0000FFFF
    x = (x | (x << 8))  & 0x00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F
    x = (x | (x << 2))  & 0x33333333
    x = (x | (x << 1))  & 0x55555555
    return x


def morton_sort_queries_batched(
    queries: torch.Tensor,
    scene_min: torch.Tensor,
    scene_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sort_perm, inv_perm)`` sorting each sample's queries by Morton code.

    ``queries`` has shape ``(B, M, D)`` with ``D`` in ``{2, 3}``.
    ``scene_min`` and ``scene_max`` have shape ``(B, D)`` (the BVH bounding box).
    Queries outside the bounding box are clamped before encoding.

    ``sort_perm`` (int64, shape ``(B, M)``) sorts spatially close queries into
    adjacent positions::

        queries_sorted = queries.gather(1, sort_perm.unsqueeze(-1).expand(-1, -1, D))

    ``inv_perm`` (int64, shape ``(B, M)``) is the inverse permutation, computed in
    O(M) via scatter (no second sort)::

        output_original_order = output_sorted.gather(1, inv_perm.unsqueeze(-1).expand(-1, -1, K))
    """
    # Fast path: fused CUDA kernel (Morton encode + CUB radix sort + scatter inverse).
    if queries.is_cuda:
        return _C.morton_sort_queries_batched(
            queries.contiguous(),
            scene_min.contiguous(),
            scene_max.contiguous(),
        )

    # CPU fallback: pure Python bit-spreading + argsort + scatter.
    D = queries.size(-1)
    span = (scene_max - scene_min).clamp(min=1e-7)           # (B, D)
    q_norm = (queries - scene_min.unsqueeze(1)) / span.unsqueeze(1)  # (B, M, D)
    q_norm = q_norm.clamp(0.0, 1.0)

    if D == 3:
        q_int = (q_norm * 1023.0).long()                     # 10 bits per axis
        codes = (
            _spread_bits_3d(q_int[..., 0])
            | (_spread_bits_3d(q_int[..., 1]) << 1)
            | (_spread_bits_3d(q_int[..., 2]) << 2)
        )
    else:
        q_int = (q_norm * 65535.0).long()                    # 16 bits per axis
        codes = (
            _spread_bits_2d(q_int[..., 0])
            | (_spread_bits_2d(q_int[..., 1]) << 1)
        )

    sort_perm = codes.argsort(dim=1)
    # Invert the permutation in O(M) via scatter â€” avoids a second O(M log M) sort.
    # inv_perm[b, sort_perm[b, i]] = i for all b, i.
    inv_perm = torch.empty_like(sort_perm)
    inv_perm.scatter_(
        1,
        sort_perm,
        torch.arange(sort_perm.size(1), device=sort_perm.device, dtype=sort_perm.dtype)
        .unsqueeze(0)
        .expand_as(sort_perm),
    )
    return sort_perm, inv_perm
