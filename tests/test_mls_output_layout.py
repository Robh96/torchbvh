import pytest
import torch
import torchbvh
from tests._support import ordering


@pytest.mark.parametrize("dim,k",[(2,4),(2,8),(2,16),(3,4),(3,8),(3,16)])
@pytest.mark.parametrize("slopes",[False,True])
def test_routed_query_major_output_values_and_gradients(dim,k,slopes):
    torch.manual_seed(219+dim+k)
    kwargs={}
    for route,count in (("true",47),("false",89)):
        kwargs[route+"_points"]=torch.rand(2,count,dim,device="cuda")
        kwargs[route+"_queries"]=torch.rand(2,53,4,dim,device="cuda",requires_grad=True)
        kwargs[route+"_features"]=torch.rand(2,count,4,8,device="cuda",requires_grad=True)
    mask=torch.rand(2,53,4,device="cuda")>.5
    def evaluate(mode):
        for t in kwargs.values(): t.grad=None
        with ordering(mode):
            out=torchbvh.conditional_mls_interpolate(mask,**kwargs,k=k,return_grad=slopes)
            outputs=out if slopes else (out,)
            sum(value.square().mean() for value in outputs).backward()
        return tuple(value.detach().clone() for value in outputs),[t.grad.detach().clone() for t in kwargs.values() if t.requires_grad]
    reference=evaluate("pack1")
    actual=evaluate("layout1")
    for a,r in zip(actual[0],reference[0]): torch.testing.assert_close(a,r,rtol=2e-5,atol=2e-5)
    for a,r in zip(actual[1],reference[1]): torch.testing.assert_close(a,r,rtol=5e-5,atol=5e-5)
