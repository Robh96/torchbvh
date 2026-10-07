import pytest
import torch
import torchbvh
from torchbvh import _conditional, _mls_routed


@pytest.mark.parametrize("dim,k", [(2,4),(2,8),(2,16),(3,4),(3,8),(3,16)])
def test_guarded_shared_coefficients_preserve_values_and_gradients(dim,k,monkeypatch):
    torch.manual_seed(141)
    monkeypatch.setattr(_conditional,"_DIRECT_FEATURE_BANKS",True)
    mask=torch.rand(2,37,3,device="cuda")>.4
    kwargs={}
    for route,count in (("true",41),("false",83)):
        kwargs[route+"_points"]=torch.rand(2,count,dim,device="cuda")
        kwargs[route+"_queries"]=torch.rand(2,37,3,dim,device="cuda",requires_grad=True)
        kwargs[route+"_features"]=torch.randn(2,count,3,16,device="cuda",requires_grad=True)
    tensors=[v for v in kwargs.values() if v.requires_grad]
    def evaluate(shared):
        monkeypatch.setattr(_mls_routed,"_SHARED_COEFFICIENTS",shared)
        for t in tensors:t.grad=None
        value=torchbvh.conditional_mls_interpolate(mask,**kwargs,k=k)
        value.square().mean().backward()
        return value.detach(),[t.grad.clone() for t in tensors]
    reference=evaluate(False);actual=evaluate(True)
    torch.testing.assert_close(actual[0],reference[0],rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[1],reference[1]):torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)


def test_collinear_geometry_uses_original_coefficient_fallback(monkeypatch):
    monkeypatch.setattr(_conditional,"_DIRECT_FEATURE_BANKS",True)
    x=torch.linspace(-1,1,65,device="cuda")
    points=torch.stack((x,x*1e-8),-1)[None].contiguous()
    q=torch.rand(1,31,2,2,device="cuda")
    features=torch.randn(1,65,2,8,device="cuda")
    mask=torch.rand(1,31,2,device="cuda")>.5
    kwargs=dict(true_points=points,false_points=points,true_queries=q,false_queries=q,true_features=features,false_features=features)
    monkeypatch.setattr(_mls_routed,"_SHARED_COEFFICIENTS",False)
    expected=torchbvh.conditional_mls_interpolate(mask,**kwargs)
    original=torchbvh._C.mls_routed_indexed_forward
    counts=[]
    def capture(*args):
        result=original(*args);counts.append(result[3]);return result
    monkeypatch.setattr(torchbvh._C,"mls_routed_indexed_forward",capture)
    monkeypatch.setattr(_mls_routed,"_SHARED_COEFFICIENTS",True)
    actual=torchbvh.conditional_mls_interpolate(mask,**kwargs)
    assert torch.equal(actual,expected)
    assert (counts[0]<0).all()
