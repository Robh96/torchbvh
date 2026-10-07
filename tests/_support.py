"""Scoped dispatch settings for regression tests of retained fallback kernels.

These contexts compare supported layouts and gradients. Production dispatch
does not import this module or require any settings changes.
"""
from contextlib import contextmanager

import torchbvh
from torchbvh import _conditional, _mls_routed


@contextmanager
def ordering(mode):
    """Select one of the four configurations exercised by the fallback tests."""
    if mode not in ("prepared", "pack1", "layout1", "policy6"):
        raise ValueError(f"Unknown regression configuration: {mode}")
    conditional_settings = {
        "_DIRECT_FEATURE_BANKS": False,
        "_PREPARED_KNN_THREADS": 0,
        "_BIN_RESOLUTION": 0,
        "_ADAPTIVE_BINS": False,
        "_FUSED_QUERY_PACK": False,
        "_EXPERIMENTAL_FAST_POLICY": mode == "policy6",
        "_FAST_POLICY_SPATIAL_BITS": 6,
    }
    routed_settings = {
        "_CHANNEL_TILE": 1,
        "_SHARED_COEFFICIENTS": False,
        "_COEFFICIENT_SAFE_RATIO": 0.05,
        "_QUERY_MAJOR_OUTPUT": mode == "layout1",
        "_AGGREGATE_SCATTER": False,
    }
    packed = mode in ("pack1", "layout1")
    if packed:
        conditional_settings.update(_DIRECT_FEATURE_BANKS=True,
                                    _BIN_RESOLUTION=64, _ADAPTIVE_BINS=True,
                                    _FUSED_QUERY_PACK=True)
    saved = [(module, name, getattr(module, name))
             for module, settings in ((_conditional, conditional_settings),
                                      (_mls_routed, routed_settings))
             for name in settings]
    original_sort = torchbvh._C.morton_sort_routed_queries_batched
    try:
        for module, settings in ((_conditional, conditional_settings),
                                 (_mls_routed, routed_settings)):
            for name, value in settings.items():
                setattr(module, name, value)
        if packed:
            torchbvh._C.morton_sort_routed_queries_batched = (
                lambda *args: torchbvh._C.morton_sort_routed_queries_narrow(*args, 18))
        yield
    finally:
        torchbvh._C.morton_sort_routed_queries_batched = original_sort
        for module, name, value in saved:
            setattr(module, name, value)
