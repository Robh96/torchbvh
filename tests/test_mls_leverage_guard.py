import pytest
import torch
import torchbvh
from torchbvh import _mls_routed
from tests._support import ordering


@pytest.mark.parametrize("kind",["random","hole","collinear","outside"])
def test_experimental_value_leverage_guard(kind):
    torch.manual_seed(583)
    points=torch.rand(2,1024,2,device="cuda")*2-1
    if kind=="hole":points/=points.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    if kind=="collinear":points[:,:,1]=0
    queries=torch.rand(2,512,4,2,device="cuda",requires_grad=True)*2-1
    queries=queries.detach().requires_grad_()
    if kind=="outside":
        queries=(queries.detach()+5).requires_grad_()
    features=torch.rand(2,1024,4,8,device="cuda",requires_grad=True)
    kwargs=dict(true_points=points,false_points=points,true_queries=queries,false_queries=queries,
                true_features=features,false_features=features)
    mask=torch.zeros(2,512,4,dtype=torch.bool,device="cuda")
    def evaluate(shared):
        queries.grad=None;features.grad=None
        with torchbvh.PointGeometry(points) as a,torchbvh.PointGeometry(points) as b,ordering('policy6'):
            _mls_routed._SHARED_COEFFICIENTS=shared
            _mls_routed._COEFFICIENT_SAFE_RATIO=-1.
            value=torchbvh.conditional_mls_interpolate(mask,**kwargs,true_geometry=a,false_geometry=b)
            value.square().mean().backward()
        return value.detach().clone(),queries.grad.detach().clone(),features.grad.detach().clone()
    reference=evaluate(False)
    actual=evaluate(True)
    torch.testing.assert_close(actual[0],reference[0],rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[1:],reference[1:]):torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
