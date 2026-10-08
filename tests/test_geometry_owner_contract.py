"""Release contracts exercised above the automatic dispatch threshold.

The independent reference differentiates a double-precision weighted affine
solve with fixed neighbours and live adaptive bandwidth.
Sparse upstream gradients isolate derivative correctness from the separately
retained many-to-one FP32 summation diagnostics.
"""
import pytest
import torch
import torchbvh


@pytest.fixture
def native_calls(monkeypatch):
    calls = []
    native = torchbvh._C.mls_geometry_forward

    def audit(*args):
        result = native(*args)
        calls.append(result[2].shape)
        return result

    monkeypatch.setattr(torchbvh._C, "mls_geometry_forward", audit)
    return calls


def fixture(dim, channels=64):
    vertices = ([[-1, -1], [-1, 1], [1, -1], [1, 1]] if dim == 2 else
                [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    points = torch.tensor(vertices, device="cuda", dtype=torch.float32).unsqueeze(0)
    torch.manual_seed(1290 + dim + channels)
    queries = (torch.rand(1, 4097, dim, device="cuda") - .5) * .25
    features = torch.randn(1, 4, channels, device="cuda") * .25
    return points, queries, features


def reference(points, queries, features):
    delta = queries[:, :, None, :] - points[:, None, :, :]
    distance = delta.square().sum(-1)
    bandwidth = distance.sort(-1).values[..., 1:2].clamp_min(1e-12)
    weights = torch.exp(-distance / (2 * bandwidth))
    basis = torch.cat((torch.ones_like(delta[..., :1]), delta), -1)
    matrix = torch.einsum("bmni,bmn,bmnj->bmij", basis, weights, basis)
    matrix = matrix + torch.eye(basis.size(-1), dtype=matrix.dtype) * 1e-6
    rhs = torch.einsum("bmni,bmn,bnc->bmic", basis, weights, features)
    solution = torch.linalg.solve(matrix, rhs)
    return solution[:, :, 0], solution[:, :, 1:]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("channels", [4, 64])
@pytest.mark.parametrize("requested", ["both", "queries", "features"])
@pytest.mark.parametrize("slopes_only", [False, True])
def test_geometry_owner_independent_derivatives(dim, channels, requested, slopes_only, native_calls):
    points, queries, features = fixture(dim, channels)
    queries.requires_grad_(requested != "features")
    features.requires_grad_(requested != "queries")
    actual = torchbvh.mls_interpolate(points, queries, features, k=4, return_grad=True)
    rq = queries.detach().cpu().double().requires_grad_(queries.requires_grad)
    rf = features.detach().cpu().double().requires_grad_(features.requires_grad)
    expected = reference(points.cpu().double(), rq, rf)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu().double(), b, rtol=2e-5, atol=2e-5)
    upstream = [torch.randn_like(a[:, :17]) * .01 for a in actual]
    losses = [sum((o[:, :17] * g.to(o)).sum() for o, g in zip(outputs, upstream)
                  if not slopes_only or o.ndim == 4) for outputs in (actual, expected)]
    actual_inputs = [t for t in (queries, features) if t.requires_grad]
    reference_inputs = [t for t in (rq, rf) if t.requires_grad]
    gradients = torch.autograd.grad(losses[0], actual_inputs)
    targets = torch.autograd.grad(losses[1], reference_inputs)
    for a, b in zip(gradients, targets):
        torch.testing.assert_close(a.cpu().double(), b, rtol=5e-5, atol=5e-5)
    assert native_calls == [torch.Size([4097, 7 if dim == 2 else 11])]


@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_owner_noncontiguous_snapshot_and_mutation(dim, native_calls):
    points, queries, features = fixture(dim)
    points = points.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
    queries = queries.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
    features = features.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
    assert not points.is_contiguous() and not queries.is_contiguous() and not features.is_contiguous()
    target = reference(points.detach().cpu().double(), queries.detach().cpu().double(), features.detach().cpu().double())
    geometry = torchbvh.PointGeometry(points)
    values, slopes = torchbvh.mls_interpolate(points, queries, features, k=4, geometry=geometry, return_grad=True)
    with torch.no_grad():
        points.add_(3)
    with pytest.raises(ValueError, match="source changed"):
        geometry.validate(points)
    geometry.destroy()
    (values[:, :17].square().mean() + slopes[:, :17].square().mean()).backward()
    for a, b in zip((values, slopes), target):
        torch.testing.assert_close(a.cpu().double(), b, rtol=2e-5, atol=2e-5)
    assert points.grad is None
    assert torch.isfinite(queries.grad).all() and torch.isfinite(features.grad).all()
    assert native_calls


@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_owner_exact_duplicates_and_inactive_nans(dim, native_calls):
    points = torch.zeros(1, 4, dim, device="cuda")
    queries = torch.zeros(1, 4097, 1, dim, device="cuda", requires_grad=True)
    features = torch.randn(1, 4, 1, 64, device="cuda", requires_grad=True)
    inactive_q = torch.full_like(queries, float("nan"), requires_grad=True)
    inactive_f = torch.full_like(features, float("nan"), requires_grad=True)
    values, slopes = torchbvh.conditional_mls_interpolate(
        torch.ones(1, 4097, 1, device="cuda", dtype=torch.bool),
        true_points=points, false_points=points, true_queries=queries,
        false_queries=inactive_q, true_features=features, false_features=inactive_f,
        return_grad=True)
    torch.testing.assert_close(values, features.mean(1, keepdim=True).expand_as(values), rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(slopes) == 0
    (values[:, :1].sum() + slopes[:, :1].sum()).backward()
    torch.testing.assert_close(features.grad, torch.full_like(features, .25), rtol=5e-5, atol=5e-5)
    assert torch.count_nonzero(queries.grad) == 0
    assert torch.count_nonzero(inactive_q.grad) == 0
    assert torch.count_nonzero(inactive_f.grad) == 0
    assert native_calls


@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_owner_degenerate_finite_and_values_only(dim, native_calls):
    points, queries, features = fixture(dim)
    points[..., 1:] = 0
    queries.requires_grad_()
    features.requires_grad_()
    values = torchbvh.mls_interpolate(points, queries, features, k=4)
    values[:, :17].square().mean().backward()
    assert torch.isfinite(values).all()
    assert torch.isfinite(queries.grad).all() and torch.isfinite(features.grad).all()
    assert native_calls


@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_owner_stream_and_cuda_graph_backward(dim, native_calls):
    points, queries, features = fixture(dim)
    queries.requires_grad_()
    features.requires_grad_()

    def evaluate():
        with torchbvh.PointGeometry(points) as geometry:
            values, slopes = torchbvh.mls_interpolate(points, queries, features, k=4, geometry=geometry, return_grad=True)
        (values[:, :17].square().mean() + slopes[:, :17].square().mean()).backward()
        return values, slopes

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            queries.grad = features.grad = None
            evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    queries.grad = features.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate()
    captured_grads = queries.grad, features.grad
    for _ in range(2):
        with torch.no_grad():
            points.mul_(1.01)
            queries.add_(.001)
            features.mul_(.99)
        graph.replay()
        observations = [t.clone() for t in (*captured, *captured_grads)]
        queries.grad = features.grad = None
        eager = evaluate()
        for i, (a, b) in enumerate(zip(observations, (*eager, queries.grad, features.grad))):
            tolerance = 2e-5 if i < 2 else 5e-5
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    assert len(native_calls) == 6


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("conditional", [False, True])
def test_geometry_owner_batched_heads_and_two_banks(dim, conditional, native_calls):
    points, base_q, base_f = fixture(dim)
    points = points.repeat(2, 1, 1)
    false_points = points + 3
    queries = base_q[:, :1025, None].repeat(2, 1, 2, 1).requires_grad_()
    false_queries = (queries.detach() + 3).requires_grad_()
    features = base_f[:, :, None].repeat(2, 1, 2, 1).requires_grad_()
    false_features = (features.detach() * .7).requires_grad_()
    mask = torch.arange(1025, device="cuda")[None, :, None].expand(2, -1, 2) % 2 == 0
    with torchbvh.PointGeometry(points) as geometry:
        if conditional:
            with torchbvh.PointGeometry(false_points) as other:
                actual = torchbvh.conditional_mls_interpolate(
                    mask, true_points=points, false_points=false_points,
                    true_queries=queries, false_queries=false_queries,
                    true_features=features, false_features=false_features,
                    true_geometry=geometry, false_geometry=other, return_grad=True)
        else:
            actual = torchbvh.bvh_mls_interpolate_batched_heads(
                points, queries, features, k=4, geometry=geometry, return_grad=True)
    inputs = [queries, features] + ([false_queries, false_features] if conditional else [])
    references = [t.detach().cpu().double().requires_grad_() for t in inputs]
    rq, rf = references[:2]
    true_outputs = [reference(points.cpu().double(), rq[:, :, h], rf[:, :, h]) for h in range(2)]
    expected = tuple(torch.stack([o[i] for o in true_outputs], 2) for i in range(2))
    if conditional:
        fq, ff = references[2:]
        false_outputs = [reference(false_points.cpu().double(), fq[:, :, h], ff[:, :, h]) for h in range(2)]
        expected = tuple(torch.where(mask.cpu().reshape(2, 1025, 2, *([1] * (a.ndim - 3))), a,
                                   torch.stack([o[i] for o in false_outputs], 2))
                         for i, a in enumerate(expected))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu().double(), b, rtol=2e-5, atol=2e-5)
    losses = [sum(o[:, :17].square().mean() for o in outputs) for outputs in (actual, expected)]
    for a, b in zip(torch.autograd.grad(losses[0], inputs), torch.autograd.grad(losses[1], references)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=5e-5, atol=5e-5)
    assert native_calls == [torch.Size([4100, 7 if dim == 2 else 11])]


@pytest.mark.parametrize("mutated", ["queries", "features"])
def test_geometry_owner_rejects_saved_tensor_mutation(mutated, native_calls):
    points, queries, features = fixture(3)
    queries.requires_grad_()
    features.requires_grad_()
    values = torchbvh.mls_interpolate(points, queries, features, k=4)
    with torch.no_grad():
        (queries if mutated == "queries" else features).add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        values[:, :17].sum().backward()
    assert native_calls
