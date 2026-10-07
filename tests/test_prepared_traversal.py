import pytest
import torch
import torchbvh


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
@pytest.mark.parametrize("threads", [64, 128, 256, 512])
def test_explicit_traversal_preserves_order_and_ties(dim, k, threads):
    torch.manual_seed(317 + k)
    points = [torch.randn(2, n, dim, device="cuda") for n in (k + 1, 137)]
    points[0][:, :k] = 0
    points[1][:, :k] = 0
    queries = torch.randn(2, 41, dim, device="cuda")
    queries[:, 0] = 0
    routes = torch.rand(2, 41, device="cuda") > .5
    order = torch.arange(41, device="cuda").expand(2, -1).contiguous()
    with torchbvh.PointGeometry(points[0]) as a, torchbvh.PointGeometry(points[1]) as b:
        aa, bb = a.bvh, b.bvh
        args = (aa["node_aabbs"], aa["sorted_indices"], bb["node_aabbs"], bb["sorted_indices"],
                queries, routes, order, aa["num_leaves"], aa["num_real_nodes"],
                bb["num_leaves"], bb["num_real_nodes"], dim, k)
        expected = torchbvh._C.query_knn_routed_batched_cached_bounds_spatial(*args)
        topology = tuple(g[key] for g in (aa, bb) for key in ("left_child_mem", "right_child_mem", "mem_to_leaf"))
        actual = torchbvh._C.query_knn_routed_explicit(*args, *topology, threads)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        for route, p in zip((True, False), points):
            # Compare distances against independent brute force, including ties.
            distances = (queries[:, :, None] - p[:, None]).square().sum(-1).sort(-1).values[:, :, :k]
            torch.testing.assert_close(actual[1][routes == route], distances[routes == route], atol=2e-6, rtol=2e-6)
