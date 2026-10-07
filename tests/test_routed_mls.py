import pytest
import torch
import torchbvh
from torchbvh import _conditional


@pytest.mark.parametrize("dim,k,channels", [(2,4,4),(2,8,8),(2,16,16),(3,4,4),(3,8,8),(3,16,16)])
def test_direct_feature_banks_preserve_slope_loss_gradients(dim, k, channels, monkeypatch):
    torch.manual_seed(717)
    mask = torch.rand(2, 19, 3, device="cuda") > .4
    kwargs = {}
    for route, count in (("true", 21), ("false", 47)):
        kwargs[route+"_points"] = torch.rand(2,count,dim,device="cuda")
        kwargs[route+"_queries"] = torch.rand(2,19,3,dim,device="cuda",requires_grad=True)
        kwargs[route+"_features"] = torch.randn(2,count,3,channels,device="cuda",requires_grad=True)
    tensors=[v for v in kwargs.values() if v.requires_grad]
    def evaluate(direct):
        monkeypatch.setattr(_conditional, "_DIRECT_FEATURE_BANKS", direct)
        for t in tensors: t.grad=None
        values, slopes = torchbvh.conditional_mls_interpolate(mask, **kwargs, k=k, return_grad=True)
        (values.square().mean()+slopes.square().mean()).backward()
        return values.detach(), slopes.detach(), [t.grad.clone() for t in tensors]
    reference=evaluate(False)
    actual=evaluate(True)
    assert torch.equal(actual[0],reference[0]) and torch.equal(actual[1],reference[1])
    for a,r in zip(actual[2],reference[2]):
        torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
