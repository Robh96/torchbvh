import pytest
import torch
import torchbvh
from torchbvh import _mls_routed
from tests._support import ordering


@pytest.mark.parametrize("channels",[4,8,16])
@pytest.mark.parametrize("kind",["random","duplicate","collinear"])
def test_warp_scatter_aggregation_gradients(channels,kind):
    torch.manual_seed(737)
    points=torch.rand(2,89,2,device="cuda")
    if kind=="duplicate": points[:,:20]=0
    if kind=="collinear": points[:,:,1]*=1e-8
    queries=torch.rand(2,113,4,2,device="cuda",requires_grad=True)
    features=torch.rand(2,89,4,channels,device="cuda",requires_grad=True)
    mask=torch.zeros((2,113,4),dtype=torch.bool,device="cuda")
    kw=dict(true_points=points,false_points=points,true_queries=queries,false_queries=queries,
            true_features=features,false_features=features)
    def evaluate(aggregate):
        queries.grad=None;features.grad=None
        with ordering("layout1"):
            _mls_routed._AGGREGATE_SCATTER=aggregate
            output=torchbvh.conditional_mls_interpolate(mask,**kw)
            output.square().mean().backward()
        return output.detach().clone(),queries.grad.detach().clone(),features.grad.detach().clone()
    reference=evaluate(False)
    actual=evaluate(True)
    for a,r in zip(actual,reference): torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
