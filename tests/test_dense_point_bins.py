import pytest
import torch
import torchbvh

from torchbvh import _conditional


@pytest.mark.parametrize("height,width", [(125, 200), (200, 250), (250, 400)])
def test_dense_fast_policy_avoids_fallback_and_preserves_bvh_values_and_gradients(height, width):
    torch.manual_seed(812)
    batch, queries, heads, channels = 2, 513, 8, 4
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device="cuda"),
                            torch.linspace(0, 1, width, device="cuda"), indexing="ij")
    points = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(batch, -1, -1).contiguous()
    corners = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], device="cuda")
    boundary = corners[None].expand(batch, -1, -1).contiguous()
    query = (.05 + .9 * torch.rand(batch, queries, heads, 2, device="cuda")).requires_grad_()
    features = torch.randn(batch, height * width, heads, channels, device="cuda", requires_grad=True)
    mask = torch.zeros(batch, queries, heads, dtype=torch.bool, device="cuda")
    kw = dict(true_points=boundary, true_queries=query.detach(),
              true_features=torch.zeros(batch, 4, heads, channels, device="cuda"),
              false_points=points, false_queries=query, false_features=features, k=4)
    upstream = torch.randn(batch, queries, heads, channels, device="cuda")
    with torchbvh.PointGeometry(boundary) as bg, torchbvh.PointGeometry(points) as fg:
        reference = torchbvh.conditional_mls_interpolate(mask, **kw, true_geometry=bg, false_geometry=fg)
        reference.backward(upstream)
    expected_features, expected_queries = features.grad.clone(), query.grad.clone()
    features.grad = query.grad = None
    flags = []
    previous = _conditional._BIN_DIAGNOSTICS
    try:
        _conditional._BIN_DIAGNOSTICS = flags
        with torchbvh.PointGeometry(boundary) as bg, torchbvh.PointGeometry(points, experimental_fast=True) as fg:
            actual = torchbvh.conditional_mls_interpolate(mask, **kw, true_geometry=bg, false_geometry=fg)
            actual.backward(upstream)
    finally:
        _conditional._BIN_DIAGNOSTICS = previous
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(features.grad, expected_features, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(query.grad, expected_queries, rtol=5e-5, atol=5e-5)
    # Pin the actual performance failure, rather than a chosen bin-size formula.
    assert len(flags) == 1
    assert flags[0].float().mean().item() < .01


@pytest.mark.parametrize("kind", ["random", "clustered", "ties", "translated", "tiny"])
def test_256_bins_preserve_exact_neighbors_and_safe_fallback(kind):
    torch.manual_seed(412)
    batch, count, queries = 2, 10000, 513
    points = torch.rand(batch, count, 2, device="cuda")
    query = torch.rand(batch, queries, 2, device="cuda") * 1.2 - .1
    if kind == "clustered":
        points *= .001
        points[:, :128] = .5
    elif kind == "ties":
        points[:, :16] = .5
        query[:, :12] = .5
    elif kind == "translated":
        points += 1e4
        query += 1e4
    elif kind == "tiny":
        points *= 1e-8
        query *= 1e-8
    other = torch.rand(batch, 17, 2, device="cuda")
    routes = torch.rand(batch, queries, device="cuda") > .9
    routes[:, :12] = False
    order = torch.arange(queries, device="cuda").expand(batch, -1).contiguous()
    with torchbvh.PointGeometry(other) as a, torchbvh.PointGeometry(points) as b:
        aa, bb = a.bvh, b.bvh
        args = (aa["node_aabbs"], aa["sorted_indices"], bb["node_aabbs"], bb["sorted_indices"],
                query, routes, order, aa["num_leaves"], aa["num_real_nodes"],
                bb["num_leaves"], bb["num_real_nodes"], 2, 4)
        expected = torchbvh._C.query_knn_routed_batched_cached_bounds_spatial(*args)
        bins = torchbvh._C.build_point_bins(points, bb["scene_min"], bb["scene_max"], 256)
        actual = torchbvh._C.query_knn_routed_bins(*args, points, bb["scene_min"], bb["scene_max"], *bins, 256)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        if kind == "ties":
            assert actual[2][:, :12].all()
