import pytest
import torch
import torchbvh


@pytest.mark.parametrize("kind", ["random", "clustered", "hole", "collinear", "ties", "tiny", "translated"])
@pytest.mark.parametrize("resolution", [16, 64, 128])
def test_bins_with_fallback_match_bvh_exactly(kind, resolution):
    torch.manual_seed(725)
    batch, count, query_count = 3, 1027, 211
    points = torch.rand(batch, count, 2, device="cuda") * 2 - 1
    queries = torch.rand(batch, query_count, 2, device="cuda") * 4 - 2
    if kind == "clustered":
        points = points.sign() * points.abs().pow(5)
    elif kind == "hole":
        points /= points.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    elif kind == "collinear":
        points[..., 1] = points[..., 0] * 1e-8
    elif kind == "ties":
        points[:, :16] = 0
        queries[:, :12] = 0
    elif kind == "tiny":
        points *= 1e-8
        queries *= 1e-8
    elif kind == "translated":
        points += 1e4
        queries += 1e4
    # Independent clouds, outside queries, and extreme offsets exercise bounds.
    points[1] += .037
    queries[:, -2] = -1e6
    queries[:, -1] = 1e6
    other = torch.rand(batch, 17, 2, device="cuda")
    routes = torch.rand(batch, query_count, device="cuda") > .85
    order = torch.arange(query_count, device="cuda").expand(batch, -1).contiguous()
    with torchbvh.PointGeometry(other) as a, torchbvh.PointGeometry(points) as b:
        aa, bb = a.bvh, b.bvh
        args = (aa["node_aabbs"], aa["sorted_indices"], bb["node_aabbs"], bb["sorted_indices"],
                queries, routes, order, aa["num_leaves"], aa["num_real_nodes"],
                bb["num_leaves"], bb["num_real_nodes"], 2, 4)
        expected = torchbvh._C.query_knn_routed_batched_cached_bounds_spatial(*args)
        bins = torchbvh._C.build_point_bins(points, bb["scene_min"], bb["scene_max"], resolution)
        actual = torchbvh._C.query_knn_routed_bins(*args, points, bb["scene_min"], bb["scene_max"], *bins, resolution)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        if kind == "ties":
            assert actual[2][:, :12].all()
