import pytest
import torch
import torchbvh
from tests._support import ordering


@pytest.mark.parametrize("query_grad,feature_grad",[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("return_grad",[False,True])
@pytest.mark.parametrize("channels",[4,8,16])
def test_routed_backward_only_requested_gradients(query_grad,feature_grad,return_grad,channels):
    torch.manual_seed(295)
    mask=torch.rand(2,113,4,device="cuda")>.5
    kwargs={}
    for route,count in (("true",47),("false",89)):
        kwargs[route+"_points"]=torch.rand(2,count,2,device="cuda")
        kwargs[route+"_queries"]=torch.rand(2,113,4,2,device="cuda",requires_grad=query_grad)
        kwargs[route+"_features"]=torch.rand(2,count,4,channels,device="cuda",requires_grad=feature_grad)
    def evaluate(mode):
        for t in kwargs.values(): t.grad=None
        with ordering(mode):
            result=torchbvh.conditional_mls_interpolate(mask,**kwargs,return_grad=return_grad)
            outputs=result if return_grad else (result,)
            sum(t.square().mean() for t in outputs).backward()
        return [t.detach().clone() for t in outputs],[t.grad.detach().clone() if t.grad is not None else None for t in kwargs.values()]
    reference=evaluate("prepared")
    actual=evaluate("layout1")
    for a,r in zip(actual[0],reference[0]): torch.testing.assert_close(a,r,rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[1],reference[1]):
        if r is None: assert a is None
        else: torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
