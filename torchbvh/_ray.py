from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from . import _C


PrimitiveType = Literal["segment", "triangle"]


@dataclass
class RayHitResult:
    """Closest-hit ray tracing result.

    Misses have ``primitive_indices == -1``, ``t == inf``, ``points == nan``,
    and ``mask == False``. Hit selection is discrete; ``t`` and ``points`` have
    piecewise gradients through the selected primitive and the live ray tensors.
    """

    primitive_indices: torch.Tensor
    t: torch.Tensor
    points: torch.Tensor
    mask: torch.Tensor


def _validate_primitives(primitives: torch.Tensor, primitive_type: str) -> tuple[bool, int]:
    if primitive_type not in ("segment", "triangle"):
        raise ValueError("RayBVH: primitive_type must be 'segment' or 'triangle'")
    if not isinstance(primitives, torch.Tensor):
        raise TypeError("RayBVH: primitives must be a torch.Tensor")
    if primitives.dim() not in (3, 4):
        raise ValueError("RayBVH: primitives must have shape (F, V, D) or (B, F, V, D)")
    expected_tail = (2, 2) if primitive_type == "segment" else (3, 3)
    if tuple(primitives.shape[-2:]) != expected_tail:
        label = "2-D segments (..., F, 2, 2)" if primitive_type == "segment" else "3-D triangles (..., F, 3, 3)"
        raise ValueError(f"RayBVH: primitive_type={primitive_type!r} requires {label}")
    if primitives.size(-3) < 1:
        raise ValueError("RayBVH: each sample must contain at least one primitive")
    if primitives.dim() == 4 and primitives.size(0) < 1:
        raise ValueError("RayBVH: batch size must be at least one")
    if primitives.dtype != torch.float32:
        raise ValueError("RayBVH: primitives must be float32")
    if not primitives.is_cuda:
        raise ValueError("RayBVH: primitives must be CUDA tensors")
    return primitives.dim() == 3, expected_tail[1]


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _selected_hit_t(
    primitives: torch.Tensor,
    origins: torch.Tensor,
    directions: torch.Tensor,
    indices: torch.Tensor,
    cuda_t: torch.Tensor,
    primitive_type: PrimitiveType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = indices >= 0
    safe_indices = indices.clamp_min(0)
    batch_indices = torch.arange(
        primitives.size(0), device=primitives.device, dtype=torch.int64
    ).view(-1, 1)
    selected = primitives[batch_indices, safe_indices]

    if primitive_type == "segment":
        edge = selected[..., 1, :] - selected[..., 0, :]
        denominator = _cross2(directions, edge)
        safe_denominator = torch.where(mask, denominator, torch.ones_like(denominator))
        live_t = _cross2(selected[..., 0, :] - origins, edge) / safe_denominator
    else:
        edge1 = selected[..., 1, :] - selected[..., 0, :]
        edge2 = selected[..., 2, :] - selected[..., 0, :]
        pvec = torch.linalg.cross(directions, edge2, dim=-1)
        determinant = (edge1 * pvec).sum(dim=-1)
        safe_determinant = torch.where(mask, determinant, torch.ones_like(determinant))
        tvec = origins - selected[..., 0, :]
        qvec = torch.linalg.cross(tvec, edge1, dim=-1)
        live_t = (edge2 * qvec).sum(dim=-1) / safe_determinant

    safe_cuda_t = torch.where(mask, cuda_t, torch.zeros_like(cuda_t))
    corrected_t = live_t + (safe_cuda_t - live_t).detach()
    output_t = torch.where(mask, corrected_t, torch.full_like(corrected_t, float("inf")))
    safe_point_t = torch.where(mask, corrected_t, torch.zeros_like(corrected_t))
    live_points = origins + safe_point_t.unsqueeze(-1) * directions
    output_points = torch.where(
        mask.unsqueeze(-1), live_points, torch.full_like(live_points, float("nan"))
    )
    return output_t, output_points, mask


class RayBVH:
    """Reusable closest-hit BVH for 2-D segments or 3-D triangles.

    Fixed-size batches build one independent BVH per sample. Geometry is treated
    as immutable for the object's lifetime; rebuild after changing its values.
    """

    def __init__(
        self,
        primitives: torch.Tensor,
        *,
        primitive_type: PrimitiveType,
    ) -> None:
        self._single, self._dim = _validate_primitives(primitives, primitive_type)
        self._primitive_type: PrimitiveType = primitive_type
        self._primitives = primitives
        self._primitives_batched = primitives.unsqueeze(0) if self._single else primitives
        build_primitives = self._primitives_batched.detach().contiguous()
        self._data: dict | None = dict(_C.build_primitive_bvh_batched(build_primitives))

    @property
    def destroyed(self) -> bool:
        return self._data is None

    def _require_live(self) -> dict:
        if self._data is None:
            raise RuntimeError("RayBVH has been destroyed")
        return self._data

    def trace(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        *,
        t_min: float = 1e-7,
        t_max: float = float("inf"),
    ) -> RayHitResult:
        """Return the exact closest hit for each ray ``origin + t * direction``."""
        data = self._require_live()
        if not isinstance(origins, torch.Tensor) or not isinstance(directions, torch.Tensor):
            raise TypeError("RayBVH.trace: origins and directions must be torch.Tensor instances")
        if origins.shape != directions.shape:
            raise ValueError("RayBVH.trace: origins and directions must have identical shapes")
        minimum_dims = 1 if self._single else 2
        if origins.dim() < minimum_dims or origins.size(-1) != self._dim:
            expected = "(..., D)" if self._single else "(B, ..., D)"
            raise ValueError(f"RayBVH.trace: rays must have shape {expected}")
        if not self._single and origins.size(0) != data["batch_size"]:
            raise ValueError("RayBVH.trace: ray batch size must match the primitive batch")
        if origins.dtype != torch.float32 or directions.dtype != torch.float32:
            raise ValueError("RayBVH.trace: origins and directions must be float32")
        if not origins.is_cuda or not directions.is_cuda:
            raise ValueError("RayBVH.trace: origins and directions must be CUDA tensors")
        if origins.device != self._primitives_batched.device or directions.device != origins.device:
            raise ValueError("RayBVH.trace: primitives, origins, and directions must share a device")

        t_min = float(t_min)
        t_max = float(t_max)
        if not math.isfinite(t_min) or math.isnan(t_max) or t_min < 0.0 or t_max <= t_min:
            raise ValueError("RayBVH.trace: expected finite 0 <= t_min < t_max")

        result_shape = tuple(origins.shape[:-1])
        if self._single:
            flat_origins = origins.reshape(1, -1, self._dim).contiguous()
            flat_directions = directions.reshape(1, -1, self._dim).contiguous()
        else:
            flat_origins = origins.reshape(data["batch_size"], -1, self._dim).contiguous()
            flat_directions = directions.reshape(data["batch_size"], -1, self._dim).contiguous()

        primitive_values = self._primitives_batched.detach().contiguous()
        indices, cuda_t = _C.raytrace_batched(
            data["node_aabbs"],
            data["sorted_indices"],
            data["left_child_mem"],
            data["right_child_mem"],
            data["mem_to_leaf"],
            primitive_values,
            flat_origins.detach(),
            flat_directions.detach(),
            data["num_real_nodes"],
            t_min,
            t_max,
        )
        output_t, points, mask = _selected_hit_t(
            self._primitives_batched,
            flat_origins,
            flat_directions,
            indices.detach(),
            cuda_t.detach(),
            self._primitive_type,
        )

        return RayHitResult(
            primitive_indices=indices.reshape(result_shape),
            t=output_t.reshape(result_shape),
            points=points.reshape((*result_shape, self._dim)),
            mask=mask.reshape(result_shape),
        )

    def destroy(self) -> None:
        if self._data is not None:
            self._data.clear()
        self._data = None
        self._primitives = None
        self._primitives_batched = None

    def __enter__(self) -> "RayBVH":
        self._require_live()
        return self

    def __exit__(self, *args) -> None:
        self.destroy()


def raytrace(
    primitives: torch.Tensor,
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    primitive_type: PrimitiveType,
    t_min: float = 1e-7,
    t_max: float = float("inf"),
) -> RayHitResult:
    """Build a temporary primitive BVH and return exact closest hits."""
    with RayBVH(primitives, primitive_type=primitive_type) as bvh:
        return bvh.trace(origins, directions, t_min=t_min, t_max=t_max)


__all__ = ["RayBVH", "RayHitResult", "raytrace"]
