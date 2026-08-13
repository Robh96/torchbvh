import pytest
import torch

import torchbvh
from torchbvh._conditional import _query_knn_routed_batched


def _inputs(dim=3, k=4, *, noncontiguous=False):
    torch.manual_seed(9100 + dim + k)
    B, M, H, C = 2, 9, 3, 2
    N_true, N_false = max(19, k + 3), max(23, k + 5)

    def shaped(shape):
        if not noncontiguous:
            return torch.randn(shape, device="cuda", dtype=torch.float32)
        storage = torch.randn((*shape[:-1], shape[-1] + 1), device="cuda", dtype=torch.float32)
        result = storage[..., : shape[-1]]
        assert not result.is_contiguous()
        return result

    true_points = shaped((B, N_true, dim))
    false_points = shaped((B, N_false, dim))
    false_points.add_(0.2)
    true_queries = shaped((B, M, H, dim))
    false_queries = shaped((B, M, H, dim)) - 0.1
    true_features = shaped((B, N_true, H, C))
    false_features = shaped((B, N_false, H, C))
    mask = (torch.rand((B, H, M), device="cuda") > 0.45).transpose(1, 2)
    if not noncontiguous:
        mask = mask.contiguous()
    return mask, true_points, true_queries, true_features, false_points, false_queries, false_features


def _call(args, *, k=4, return_grad=False):
    mask, tp, tq, tf, fp, fq, ff = args
    return torchbvh.conditional_mls_interpolate(
        mask,
        true_points=tp,
        true_queries=tq,
        true_features=tf,
        false_points=fp,
        false_queries=fq,
        false_features=ff,
        k=k,
        return_grad=return_grad,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_conditional_matches_two_mls_branches(dim, k):
    args = _inputs(dim, k)
    mask, tp, tq, tf, fp, fq, ff = args
    actual, actual_grad = _call(args, k=k, return_grad=True)
    true_value, true_grad = torchbvh.bvh_mls_interpolate_batched_heads(
        tp, tq, tf, k=k, return_grad=True)
    false_value, false_grad = torchbvh.bvh_mls_interpolate_batched_heads(
        fp, fq, ff, k=k, return_grad=True)
    expected = torch.where(mask[..., None], true_value, false_value)
    expected_grad = torch.where(mask[..., None, None], true_grad, false_grad)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)
    assert actual.shape == (*mask.shape, tf.size(-1))
    assert actual_grad.shape == (*mask.shape, dim, tf.size(-1))


@pytest.mark.parametrize("value", [False, True])
def test_conditional_all_one_route(value):
    args = list(_inputs(2, 4))
    args[0] = torch.full_like(args[0], value)
    actual = _call(tuple(args))
    point_index, query_index, feature_index = (1, 2, 3) if value else (4, 5, 6)
    expected = torchbvh.bvh_mls_interpolate_batched_heads(
        args[point_index], args[query_index], args[feature_index], k=4)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_conditional_gradients_are_routed_and_points_are_detached():
    args = list(_inputs(3, 4))
    mask = args[0]
    for index in range(1, 7):
        args[index] = args[index].requires_grad_()
    output = _call(tuple(args))
    output.square().sum().backward()

    tp, tq, tf, fp, fq, ff = args[1:]
    assert tp.grad is None and fp.grad is None
    assert tq.grad is not None and fq.grad is not None
    assert tf.grad is not None and ff.grad is not None
    assert torch.count_nonzero(tq.grad[~mask]) == 0
    assert torch.count_nonzero(fq.grad[mask]) == 0
    assert torch.count_nonzero(tq.grad[mask]) > 0
    assert torch.count_nonzero(fq.grad[~mask]) > 0


@pytest.mark.parametrize("value", [False, True])
def test_conditional_inactive_query_and_feature_branch_has_zero_gradient(value):
    args = list(_inputs(2, 4))
    args[0] = torch.full_like(args[0], value)
    for index in (2, 3, 5, 6):
        args[index] = args[index].requires_grad_()
    _call(tuple(args)).sum().backward()
    inactive_query = args[5] if value else args[2]
    inactive_features = args[6] if value else args[3]
    assert inactive_query.grad is not None
    assert inactive_features.grad is not None
    assert torch.count_nonzero(inactive_query.grad) == 0
    assert torch.count_nonzero(inactive_features.grad) == 0


def test_conditional_duplicate_exact_hit_neighborhoods():
    args = list(_inputs(3, 4))
    mask, tp, tq, tf, fp, fq, ff = args
    tp = tp.clone()
    fp = fp.clone()
    tq = tq.clone()
    fq = fq.clone()
    tp[:, :4] = 0
    fp[:, :4] = 0
    tq[:] = 0
    fq[:] = 0
    args[1], args[2], args[4], args[5] = tp, tq, fp, fq
    values, gradient = _call(tuple(args), return_grad=True)
    true_mean = tf[:, :4].mean(dim=1)[:, None].expand_as(values)
    false_mean = ff[:, :4].mean(dim=1)[:, None].expand_as(values)
    expected = torch.where(mask[..., None], true_mean, false_mean)
    torch.testing.assert_close(values, expected)
    assert torch.count_nonzero(gradient) == 0


def test_conditional_inactive_nan_queries_are_not_traversed():
    args = list(_inputs(3, 4))
    mask = args[0]
    args[2] = args[2].clone()
    args[5] = args[5].clone()
    args[2][~mask] = torch.nan
    args[5][mask] = torch.nan
    result = _call(tuple(args))
    assert torch.isfinite(result).all()


def test_conditional_accepts_noncontiguous_inputs():
    args = _inputs(2, 4, noncontiguous=True)
    result = _call(args)
    assert result.shape == (*args[0].shape, args[3].size(-1))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_private_routed_knn_matches_exact_branch_queries_and_offsets(dim):
    args = _inputs(dim, 4)
    mask, tp, tq, _, fp, fq, _ = args
    B, M, H = mask.shape
    selected = torch.where(mask[..., None], tq, fq).permute(0, 2, 1, 3).reshape(B, H * M, dim).contiguous()
    routes = mask.permute(0, 2, 1).reshape(B, H * M).contiguous()
    true_bvh = torchbvh.build_bvh_batched(tp.contiguous())
    false_bvh = torchbvh.build_bvh_batched(fp.contiguous())
    try:
        indices, distances = _query_knn_routed_batched(true_bvh, false_bvh, selected, routes, 4)
    finally:
        torchbvh.destroy_bvh(true_bvh)
        torchbvh.destroy_bvh(false_bvh)

    true_dist, true_idx = torch.cdist(selected, tp).square().topk(4, largest=False)
    false_dist, false_idx = torch.cdist(selected, fp).square().topk(4, largest=False)
    expected_dist = torch.where(routes[..., None], true_dist, false_dist)
    expected_idx = torch.where(routes[..., None], true_idx, false_idx + tp.size(1))
    torch.testing.assert_close(distances, expected_dist, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(indices, expected_idx)


def test_private_routed_knn_rejects_destroyed_or_incompatible_handles():
    args = _inputs(2, 4)
    mask, tp, tq, _, fp, _, _ = args
    queries = tq.permute(0, 2, 1, 3).reshape(2, -1, 2).contiguous()
    routes = mask.permute(0, 2, 1).reshape(2, -1).contiguous()
    true_bvh = torchbvh.build_bvh_batched(tp.contiguous())
    false_bvh = torchbvh.build_bvh_batched(fp.contiguous())
    torchbvh.destroy_bvh(true_bvh)
    with pytest.raises(RuntimeError, match="destroyed"):
        _query_knn_routed_batched(true_bvh, false_bvh, queries, routes, 4)
    torchbvh.destroy_bvh(false_bvh)

    true_bvh = torchbvh.build_bvh_batched(tp.contiguous())
    wrong_false = torchbvh.build_bvh_batched(
        torch.randn((2, fp.size(1), 3), device="cuda"))
    try:
        with pytest.raises(RuntimeError, match="scene bound"):
            _query_knn_routed_batched(true_bvh, wrong_false, queries, routes, 4)
    finally:
        torchbvh.destroy_bvh(true_bvh)
        torchbvh.destroy_bvh(wrong_false)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda a: a.__setitem__(0, a[0].float()), "mask must be bool"),
        (lambda a: a.__setitem__(2, a[2][..., :1]), "query branches"),
        (lambda a: a.__setitem__(3, a[3][..., :1]), "feature branches must share C"),
    ],
)
def test_conditional_validation_errors(mutation, message):
    args = list(_inputs(2, 4))
    mutation(args)
    with pytest.raises(ValueError, match=message):
        _call(tuple(args))


def test_conditional_is_publicly_exported():
    assert "conditional_mls_interpolate" in torchbvh.__all__
    assert callable(torchbvh.conditional_mls_interpolate)
