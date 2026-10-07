import torch
import torchbvh


def test_prepared_geometry_nondefault_stream_lifetime():
    points=torch.rand(2,113,2,device="cuda")
    queries=torch.rand(2,37,2,device="cuda",requires_grad=True)
    features=torch.rand(2,113,4,device="cuda",requires_grad=True)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torchbvh.PointGeometry(points) as geometry:
            value=torchbvh.mls_interpolate(points,queries,features,geometry=geometry)
        # Saved snapshot must remain valid after geometry releases its tree.
        value.square().mean().backward()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.isfinite(value).all() and torch.isfinite(queries.grad).all()


def test_prepared_geometry_cuda_graph_rebuilds_positions():
    points=torch.rand(2,113,2,device="cuda")
    queries=torch.rand(2,37,2,device="cuda")
    features=torch.rand(2,113,4,device="cuda")
    def evaluate():
        with torchbvh.PointGeometry(points) as geometry:
            return torchbvh.mls_interpolate(points,queries,features,geometry=geometry)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output=evaluate()
    for _ in range(2):
        points.copy_(torch.rand_like(points))
        graph.replay()
        torch.testing.assert_close(output,evaluate(),rtol=2e-5,atol=2e-5)
