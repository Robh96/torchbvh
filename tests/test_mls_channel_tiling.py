import pytest
import torch
import torchbvh
from torchbvh import _conditional, _mls_routed


@pytest.mark.parametrize("dim,k,channels", [(2,4,4),(2,8,8),(2,16,16),(3,4,4),(3,8,8),(3,16,16)])
@pytest.mark.parametrize("tile", [4,8,16])
def test_tiled_routed_solve_and_gradients(dim, k, channels, tile, monkeypatch):
    torch.manual_seed(933)
    monkeypatch.setattr(_conditional, "_DIRECT_FEATURE_BANKS", True)
    mask = torch.rand(2, 27, 3, device="cuda") > .4
    kwargs = {}
    for route, count in (("true", 31), ("false", 67)):
        points = torch.rand(2,count,dim,device="cuda")
        points[:, :k] = 0
        q = torch.rand(2,27,3,dim,device="cuda")
        q[:,0] = 0
        kwargs[route+"_points"] = points
        kwargs[route+"_queries"] = q.requires_grad_()
        kwargs[route+"_features"] = torch.randn(2,count,3,channels,device="cuda",requires_grad=True)
    tensors=[v for v in kwargs.values() if v.requires_grad]
    def evaluate(channel_tile):
        monkeypatch.setattr(_mls_routed, "_CHANNEL_TILE", channel_tile)
        for t in tensors: t.grad=None
        values, slopes = torchbvh.conditional_mls_interpolate(mask, **kwargs, k=k, return_grad=True)
        (values.square().mean()+slopes.square().mean()).backward()
        return values.detach(),slopes.detach(),[t.grad.clone() for t in tensors]
    reference=evaluate(1)
    actual=evaluate(tile)
    torch.testing.assert_close(actual[0],reference[0],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(actual[1],reference[1],rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[2],reference[2]):
        torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
