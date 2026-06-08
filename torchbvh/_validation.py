import torch

from ._constants import SUPPORTED_DIMS, SUPPORTED_K


def _validate_supported_k(prefix: str, k: int) -> None:
    if k not in SUPPORTED_K:
        raise ValueError(f"{prefix}: k must be 4, 8, or 16")


def _as_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _as_contiguous_int64(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int64:
        return tensor
    return _as_contiguous(tensor)


def _validate_cuda_float32(prefix: str, tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{prefix}: {name} must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{prefix}: {name} must be float32")


def _validate_cuda_float32_contiguous(prefix: str, tensor: torch.Tensor, name: str) -> None:
    _validate_cuda_float32(prefix, tensor, name)
    if not tensor.is_contiguous():
        raise ValueError(f"{prefix}: {name} must be contiguous")


def _validate_offsets(
    prefix: str,
    offsets: torch.Tensor,
    *,
    total_rows: int,
    device: torch.device,
) -> list[int]:
    if offsets.dim() != 1:
        raise ValueError(f"{prefix}: offsets must have shape (B + 1,)")
    if offsets.numel() < 2:
        raise ValueError(f"{prefix}: offsets must contain at least two entries")
    if offsets.dtype != torch.int64:
        raise ValueError(f"{prefix}: offsets must be int64")
    if not offsets.is_cuda:
        raise ValueError(f"{prefix}: offsets must be a CUDA tensor")
    if not offsets.is_contiguous():
        raise ValueError(f"{prefix}: offsets must be contiguous")
    if offsets.device != device:
        raise ValueError(f"{prefix}: offsets must be on the same device as the packed tensor")

    values = [int(v) for v in offsets.detach().cpu().tolist()]
    if values[0] != 0:
        raise ValueError(f"{prefix}: offsets must start at 0")
    if values[-1] != total_rows:
        raise ValueError(f"{prefix}: final offset must match the packed tensor length")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{prefix}: offsets must be strictly increasing")
    return values


def _validate_ragged_points(
    prefix: str,
    points: torch.Tensor,
    offsets: torch.Tensor,
) -> list[int]:
    if points.dim() != 2:
        raise ValueError(f"{prefix}: points must have shape (total_N, D)")
    if points.size(1) not in SUPPORTED_DIMS:
        raise ValueError(f"{prefix}: D must be 2 or 3")
    _validate_cuda_float32_contiguous(prefix, points, "points")
    return _validate_offsets(prefix, offsets, total_rows=int(points.size(0)), device=points.device)


def _real_nodes_at_level(num_leaves: int, level: int) -> int:
    leaf_level = (num_leaves - 1).bit_length()
    virtual_leaves = (1 << leaf_level) - num_leaves
    return (1 << level) - (virtual_leaves >> (leaf_level - level))


def _choose_level_for_target_tokens(num_leaves: int, requested_tokens: int) -> int:
    leaf_level = (num_leaves - 1).bit_length()
    best_level = 0
    best_count = _real_nodes_at_level(num_leaves, 0)
    best_distance = abs(best_count - requested_tokens)
    for level in range(1, leaf_level + 1):
        count = _real_nodes_at_level(num_leaves, level)
        distance = abs(count - requested_tokens)
        if distance < best_distance or (distance == best_distance and count > best_count):
            best_level = level
            best_count = count
            best_distance = distance
    return best_level

