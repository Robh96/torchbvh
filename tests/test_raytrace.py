import pytest
import torch

import torchbvh


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dense_segments(primitives, origins, directions, t_min=1e-7, t_max=float("inf")):
    edge = primitives[:, None, :, 1] - primitives[:, None, :, 0]
    offset = primitives[:, None, :, 0] - origins[:, :, None]
    ray = directions[:, :, None, :]
    denominator = _cross2(ray, edge)
    tolerance = 8.0 * torch.finfo(torch.float32).eps * (
        ray.square().sum(-1) * edge.square().sum(-1)
    ).sqrt()
    valid_denominator = denominator.abs() > tolerance
    safe_denominator = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
    t = _cross2(offset, edge) / safe_denominator
    u = _cross2(offset, ray) / safe_denominator
    valid = valid_denominator & (t >= t_min) & (t <= t_max) & (u >= 0) & (u <= 1)
    candidate = torch.where(valid, t, torch.full_like(t, float("inf")))
    best_t, indices = candidate.min(dim=-1)
    indices = torch.where(torch.isfinite(best_t), indices, torch.full_like(indices, -1))
    return indices, best_t


def _dense_triangles(primitives, origins, directions, t_min=1e-7, t_max=float("inf")):
    vertex0 = primitives[:, None, :, 0]
    edge1 = primitives[:, None, :, 1] - vertex0
    edge2 = primitives[:, None, :, 2] - vertex0
    ray = directions[:, :, None, :]
    pvec = torch.linalg.cross(ray, edge2, dim=-1)
    determinant = (edge1 * pvec).sum(-1)
    normal = torch.linalg.cross(edge1, edge2, dim=-1)
    tolerance = 8.0 * torch.finfo(torch.float32).eps * (
        ray.square().sum(-1) * normal.square().sum(-1)
    ).sqrt()
    valid_determinant = determinant.abs() > tolerance
    inverse = torch.where(valid_determinant, determinant.reciprocal(), torch.zeros_like(determinant))
    tvec = origins[:, :, None, :] - vertex0
    u = (tvec * pvec).sum(-1) * inverse
    qvec = torch.linalg.cross(tvec, edge1, dim=-1)
    v = (ray * qvec).sum(-1) * inverse
    t = (edge2 * qvec).sum(-1) * inverse
    epsilon = 8.0 * torch.finfo(torch.float32).eps
    valid = (
        valid_determinant
        & (u >= -epsilon)
        & (v >= -epsilon)
        & (u + v <= 1 + epsilon)
        & (t >= t_min)
        & (t <= t_max)
    )
    candidate = torch.where(valid, t, torch.full_like(t, float("inf")))
    best_t, indices = candidate.min(dim=-1)
    indices = torch.where(torch.isfinite(best_t), indices, torch.full_like(indices, -1))
    return indices, best_t


def test_segment_raytrace_preserves_single_multihead_shape_and_miss_contract():
    primitives = torch.tensor(
        [[[1.0, -1.0], [1.0, 1.0]], [[3.0, -1.0], [3.0, 1.0]], [[5.0, -1.0], [5.0, 1.0]]],
        device="cuda",
    )
    origins = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0]], [[4.0, 0.0], [0.0, 2.0]]], device="cuda"
    )
    directions = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]], device="cuda"
    )

    result = torchbvh.raytrace(
        primitives, origins, directions, primitive_type="segment", t_max=2.0
    )

    assert result.primitive_indices.shape == (2, 2)
    assert result.points.shape == (2, 2, 2)
    torch.testing.assert_close(
        result.primitive_indices,
        torch.tensor([[0, 1], [2, -1]], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(result.t[result.mask], torch.ones(3, device="cuda"))
    assert torch.isinf(result.t[~result.mask]).all()
    assert torch.isnan(result.points[~result.mask]).all()


@pytest.mark.parametrize("primitive_type", ["segment", "triangle"])
def test_random_batched_raytrace_matches_dense_reference(primitive_type):
    torch.manual_seed(8102 if primitive_type == "segment" else 8103)
    if primitive_type == "segment":
        primitives = torch.rand((3, 37, 2, 2), device="cuda") * 4 - 2
        origins = torch.rand((3, 5, 7, 2), device="cuda") * 4 - 2
        directions = torch.rand((3, 5, 7, 2), device="cuda") * 2 - 1
        reference = _dense_segments
    else:
        primitives = torch.rand((3, 29, 3, 3), device="cuda") * 4 - 2
        origins = torch.rand((3, 5, 7, 3), device="cuda") * 4 - 2
        directions = torch.rand((3, 5, 7, 3), device="cuda") * 2 - 1
        reference = _dense_triangles

    result = torchbvh.raytrace(
        primitives, origins, directions, primitive_type=primitive_type, t_max=3.0
    )
    expected_idx, expected_t = reference(
        primitives, origins.reshape(3, -1, origins.size(-1)),
        directions.reshape(3, -1, directions.size(-1)), t_max=3.0
    )
    expected_idx = expected_idx.reshape(result.primitive_indices.shape)
    expected_t = expected_t.reshape(result.t.shape)

    torch.testing.assert_close(result.primitive_indices, expected_idx)
    torch.testing.assert_close(result.t[result.mask], expected_t[result.mask], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        result.points[result.mask],
        (origins + result.t.nan_to_num(posinf=0).unsqueeze(-1) * directions)[result.mask],
    )


def test_triangle_is_double_sided_and_respects_ray_range():
    triangle = torch.tensor(
        [[[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]], device="cuda"
    )
    origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], device="cuda")
    directions = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], device="cuda")

    result = torchbvh.raytrace(
        triangle, origins, directions, primitive_type="triangle", t_max=3.0
    )
    torch.testing.assert_close(result.primitive_indices, torch.zeros(2, device="cuda", dtype=torch.int64))
    torch.testing.assert_close(result.t, torch.tensor([1.0, 2.0], device="cuda"))

    clipped = torchbvh.raytrace(
        triangle, origins, directions, primitive_type="triangle", t_max=0.75
    )
    assert not clipped.mask.any()


def test_triangle_ties_degeneracy_and_zero_direction():
    triangles = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]],
            [[-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]],
        ],
        device="cuda",
    )
    result = torchbvh.raytrace(
        triangles,
        torch.zeros((2, 3), device="cuda"),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], device="cuda"),
        primitive_type="triangle",
    )
    assert result.primitive_indices.tolist() == [1, -1]


def test_ties_use_lowest_original_index_and_degenerate_primitives_are_ignored():
    primitives = torch.tensor(
        [
            [[2.0, -1.0], [2.0, 1.0]],
            [[2.0, -2.0], [2.0, 2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [3.0, 0.0]],
        ],
        device="cuda",
    )
    result = torchbvh.raytrace(
        primitives,
        torch.tensor([[0.0, 0.5], [0.0, 3.0], [2.0, 0.0]], device="cuda"),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], device="cuda"),
        primitive_type="segment",
    )
    assert result.primitive_indices.tolist() == [0, -1, -1]


def test_raytrace_piecewise_gradients_match_live_selected_segment_equation():
    primitives = torch.tensor(
        [[[2.0, -1.0], [2.0, 1.0]], [[4.0, -1.0], [4.0, 1.0]]],
        device="cuda", requires_grad=True,
    )
    origins = torch.tensor([[0.2, 0.25], [0.2, 3.0]], device="cuda", requires_grad=True)
    directions = torch.tensor([[1.5, 0.1], [-1.0, 0.0]], device="cuda", requires_grad=True)
    result = torchbvh.raytrace(
        primitives, origins, directions, primitive_type="segment", t_max=4.0
    )
    loss = result.t[result.mask].sum() + result.points[result.mask].sum()
    loss.backward()
    actual = (primitives.grad.clone(), origins.grad.clone(), directions.grad.clone())

    reference_primitives = primitives.detach().clone().requires_grad_()
    reference_origins = origins.detach().clone().requires_grad_()
    reference_directions = directions.detach().clone().requires_grad_()
    edge = reference_primitives[0, 1] - reference_primitives[0, 0]
    reference_t = _cross2(reference_primitives[0, 0] - reference_origins[0], edge) / _cross2(
        reference_directions[0], edge
    )
    reference_point = reference_origins[0] + reference_t * reference_directions[0]
    (reference_t + reference_point.sum()).backward()

    torch.testing.assert_close(actual[0], reference_primitives.grad)
    torch.testing.assert_close(actual[1], reference_origins.grad)
    torch.testing.assert_close(actual[2], reference_directions.grad)


def test_raytrace_piecewise_triangle_gradients_match_moller_trumbore():
    triangle = torch.tensor(
        [[[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]],
        device="cuda", requires_grad=True,
    )
    origin = torch.tensor([[0.1, 0.1, 0.0]], device="cuda", requires_grad=True)
    direction = torch.tensor([[0.05, -0.02, 1.0]], device="cuda", requires_grad=True)
    result = torchbvh.raytrace(
        triangle, origin, direction, primitive_type="triangle"
    )
    (result.t.sum() + result.points.sum()).backward()
    actual = (triangle.grad.clone(), origin.grad.clone(), direction.grad.clone())

    ref_triangle = triangle.detach().clone().requires_grad_()
    ref_origin = origin.detach().clone().requires_grad_()
    ref_direction = direction.detach().clone().requires_grad_()
    edge1 = ref_triangle[0, 1] - ref_triangle[0, 0]
    edge2 = ref_triangle[0, 2] - ref_triangle[0, 0]
    pvec = torch.linalg.cross(ref_direction[0], edge2)
    determinant = (edge1 * pvec).sum()
    tvec = ref_origin[0] - ref_triangle[0, 0]
    qvec = torch.linalg.cross(tvec, edge1)
    ref_t = (edge2 * qvec).sum() / determinant
    ref_point = ref_origin[0] + ref_t * ref_direction[0]
    (ref_t + ref_point.sum()).backward()

    torch.testing.assert_close(actual[0], ref_triangle.grad)
    torch.testing.assert_close(actual[1], ref_origin.grad)
    torch.testing.assert_close(actual[2], ref_direction.grad)


def test_ray_bvh_lifecycle_noncontiguous_inputs_and_api_exports():
    primitive_base = torch.tensor(
        [[[[1.0, 1.0], [-1.0, 1.0]], [[3.0, 3.0], [-1.0, 1.0]]]], device="cuda"
    )
    primitives = primitive_base.transpose(-1, -2)
    ray_base = torch.tensor([[[0.0, 2.0], [0.0, 0.0]]], device="cuda")
    origins = ray_base.transpose(-1, -2)
    directions = torch.ones_like(origins)
    directions[..., 1] = 0.0
    assert not primitives.is_contiguous()
    assert not origins.is_contiguous()

    bvh = torchbvh.RayBVH(primitives, primitive_type="segment")
    result = bvh.trace(origins, directions)
    assert result.primitive_indices.tolist() == [[0, 1]]
    bvh.destroy()
    bvh.destroy()
    assert bvh.destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        bvh.trace(origins, directions)

    assert {"RayBVH", "RayHitResult", "raytrace"} <= set(torchbvh.__all__)


def test_raytrace_rejects_invalid_primitive_and_batch_contracts():
    segments = torch.rand((2, 5, 2, 2), device="cuda")
    with pytest.raises(ValueError, match="primitive_type"):
        torchbvh.RayBVH(segments, primitive_type="capsule")
    with pytest.raises(ValueError, match="requires"):
        torchbvh.RayBVH(segments, primitive_type="triangle")

    bvh = torchbvh.RayBVH(segments, primitive_type="segment")
    with pytest.raises(ValueError, match="batch size"):
        bvh.trace(torch.rand((3, 4, 2), device="cuda"), torch.rand((3, 4, 2), device="cuda"))
    with pytest.raises(ValueError, match="0 <= t_min"):
        bvh.trace(torch.rand((2, 4, 2), device="cuda"), torch.rand((2, 4, 2), device="cuda"), t_min=1, t_max=1)
