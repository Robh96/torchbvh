import torch
import torchbvh


def test_explicit_fast_geometry_policy_lifetime_and_inactive_queries():
    torch.manual_seed(523)
    points=torch.rand(4,625,2,device="cuda")
    boundary=torch.rand(4,99,2,device="cuda")
    mask=torch.rand(4,724,8,device="cuda")<.2
    kwargs={"true_points":boundary,"false_points":points}
    for route,n in (("true",99),("false",625)):
        q=torch.rand(4,724,8,2,device="cuda")
        q[~mask if route=="true" else mask]=float("nan")
        kwargs[route+"_queries"]=q.requires_grad_()
        kwargs[route+"_features"]=torch.rand(4,n,8,16,device="cuda",requires_grad=True)
    def backward(geometries=None):
        for t in kwargs.values(): t.grad=None
        kw=dict(kwargs)
        if geometries: kw.update(true_geometry=geometries[0],false_geometry=geometries[1])
        value=torchbvh.conditional_mls_interpolate(mask,**kw)
        if geometries:
            for g in geometries: g.destroy()
        value.square().mean().backward()
        return value.detach().clone(),[t.grad.detach().clone() for t in kwargs.values() if t.requires_grad]
    reference=backward()
    actual=backward((torchbvh.PointGeometry(boundary),torchbvh.PointGeometry(points,experimental_fast=True)))
    torch.testing.assert_close(actual[0],reference[0],rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[1],reference[1]): torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)


def test_fast_policy_cuda_graph_rebuilds_all_geometry():
    points=torch.rand(4,625,2,device="cuda")
    boundary=torch.rand(4,99,2,device="cuda")
    mask=torch.rand(4,724,8,device="cuda")<.2
    kwargs=dict(true_points=boundary,false_points=points,
        true_queries=torch.rand(4,724,8,2,device="cuda"),
        false_queries=torch.rand(4,724,8,2,device="cuda"),
        true_features=torch.rand(4,99,8,4,device="cuda"),
        false_features=torch.rand(4,625,8,4,device="cuda"))
    def evaluate():
        with torchbvh.PointGeometry(boundary) as a, torchbvh.PointGeometry(points,experimental_fast=True) as b:
            return torchbvh.conditional_mls_interpolate(mask,**kwargs,true_geometry=a,false_geometry=b)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output=evaluate()
    for _ in range(2):
        points.copy_(torch.rand_like(points))
        boundary.copy_(torch.rand_like(boundary))
        graph.replay()
        torch.testing.assert_close(output,evaluate(),rtol=2e-5,atol=2e-5)
