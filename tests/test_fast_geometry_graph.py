"""The opt-in path must rebuild bins/BVHs and gradients on graph replay."""
import torch
import torchbvh


def test_fast_geometry_graph_refreshes_sources_queries_features_and_routes():
    torch.manual_seed(702)
    points = torch.rand(2, 1024, 2, device='cuda')
    queries = torch.rand(2, 512, 4, 2, device='cuda', requires_grad=True)
    features = torch.rand(2, 1024, 4, 4, device='cuda', requires_grad=True)
    mask = torch.rand(2, 512, 4, device='cuda') > .5

    def evaluate(fast):
        with torchbvh.PointGeometry(points, experimental_fast=fast) as geometry:
            value = torchbvh.conditional_mls_interpolate(
                mask, true_points=points, false_points=points,
                true_queries=queries, false_queries=queries,
                true_features=features, false_features=features,
                true_geometry=geometry, false_geometry=geometry)
        value.square().mean().backward()
        return value

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            queries.grad = features.grad = None
            evaluate(True)
    torch.cuda.current_stream().wait_stream(stream)
    queries.grad = features.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate(True)
    captured_query_grad = queries.grad
    captured_feature_grad = features.grad
    for _ in range(2):
        with torch.no_grad():
            points.copy_(torch.rand_like(points))
            queries.copy_(torch.rand_like(queries))
            features.copy_(torch.rand_like(features))
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > .5)
        graph.replay()
        actual = (captured.clone(), captured_query_grad.clone(), captured_feature_grad.clone())
        queries.grad = features.grad = None
        reference_value = evaluate(False)
        reference = (reference_value, queries.grad, features.grad)
        for i, (a, r) in enumerate(zip(actual, reference)):
            tolerance = 2e-5 if i == 0 else 5e-5
            torch.testing.assert_close(a, r, rtol=tolerance, atol=tolerance)
