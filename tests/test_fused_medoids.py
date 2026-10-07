import pytest
import torch
import torchbvh


@pytest.mark.parametrize("dim",[2,3])
@pytest.mark.parametrize("population",[1,2,3,4])
@pytest.mark.parametrize("kind",["random","ties","translated"])
def test_fused_medoid_centroid_reduction_and_ties(dim,population,kind):
    torch.manual_seed(145+dim)
    B,N,G=3,1027,137
    points=torch.randn(B,N,dim,device="cuda")
    if kind=="ties": points[:,::3]=0
    if kind=="translated": points+=10000
    ranks=torch.randint(N,(G,population),device="cuda")
    counts=torch.randint(1,population+1,(G,),device="cuda")
    valid=torch.arange(population,device="cuda")[None]<counts[:,None]
    with torchbvh.PointGeometry(points) as geometry:
        order=geometry.bvh["sorted_indices"]
        members=order[:,ranks]
        selected=points.gather(1,members.reshape(B,-1)[...,None].expand(-1,-1,dim)).reshape(B,G,population,dim)
        center=(selected*valid[None,...,None]).sum(2)/counts[None,:,None]
        slot=(selected-center[:,:,None]).square().sum(-1).masked_fill(~valid[None],float("inf")).argmin(2,keepdim=True)
        expected=torch.where(counts[None]==2,members[:,:,0],members.gather(2,slot).squeeze(2))
        actual,centroid=torchbvh._C.select_bvh_medoids(points,order,ranks,valid,counts)
    assert torch.equal(actual,expected)
    assert torch.equal(centroid,center)


def test_two_point_morton_rank_rule_is_explicit():
    points=torch.tensor([[[0.,0.],[1.,1.],[2.,2.],[3.,3.]]],device="cuda")
    sorted_indices=torch.tensor([[3,2,1,0]],device="cuda")
    ranks=torch.tensor([[0,1]],device="cuda")
    valid=torch.ones((1,2),dtype=torch.bool,device="cuda")
    counts=torch.tensor([2],device="cuda")
    medoid,_=torchbvh._C.select_bvh_medoids(points,sorted_indices,ranks,valid,counts)
    assert medoid.item()==3
