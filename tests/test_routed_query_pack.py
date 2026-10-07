import pytest
import torch
import torchbvh
from torchbvh._conditional_layout import _pack_routed_queries


@pytest.mark.parametrize("dim", [2,3])
@pytest.mark.parametrize("need_true,need_false", [(True,True),(True,False),(False,True)])
def test_fused_route_pack_matches_where_forward_and_backward(dim, need_true, need_false):
    torch.manual_seed(22)
    mask=torch.rand(2,31,5,device="cuda")>.4
    a=torch.randn(2,31,5,dim,device="cuda")
    b=torch.randn_like(a)
    a[~mask]=float("nan");b[mask]=float("nan")
    a.requires_grad_(need_true);b.requires_grad_(need_false)
    expected=torch.where(mask[...,None],a,b).permute(0,2,1,3).contiguous()
    actual,routes=_pack_routed_queries(a,b,mask)
    assert torch.equal(expected,actual)
    assert torch.equal(routes,mask.permute(0,2,1).reshape(2,-1))
    gradient=torch.randn_like(actual)
    tensors=[t for t in (a,b) if t.requires_grad]
    ref=torch.autograd.grad(expected,tensors,gradient,retain_graph=True)
    result=torch.autograd.grad(actual,tensors,gradient)
    for x,y in zip(ref,result): assert torch.equal(x,y)
