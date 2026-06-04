# Compatibility re-export layer. Keep imports from torchbvh.ops working
# while implementation lives in responsibility-specific private modules.

from ._constants import (
    EXACT_DISTANCE_EPSILON,
    MLS_BANDWIDTH_MIN,
    MLS_REGULARIZATION,
    SUPPORTED_DIMS,
    SUPPORTED_K,
)
from ._bvh_class import (
    BVH,
    BatchedBVH,
    RaggedBVH,
)
from ._handles import (
    BVHHandle,
    BatchedBVHHandle,
    RaggedBVHHandle,
    destroy_bvh,
)
from ._fps import (
    FPSResult,
    fps,
)
from ._mls import (
    BVHQuery,
    BatchedBVHQuery,
    bvh_mls_interpolate,
    bvh_mls_interpolate_batched,
    mls_interpolate,
)
from ._multihead import (
    gather_neighbor_values,
    interpolate_displaced,
    query_displaced_knn,
)
from ._query import (
    build_bvh,
    build_bvh_batched,
    build_bvh_ragged,
    query_knn,
    query_knn_batched,
    query_knn_ragged,
)

# Test/debug arithmetic wrappers stay separate from stable runtime and prototypes.
from ._tree import (
    implicit_tree_ancestor,
    implicit_tree_descendant,
    implicit_tree_summary,
    morton_encode_2d,
    morton_encode_3d,
    morton_split2,
    morton_split3,
    smoke_add_one,
)

__all__ = [
    # Class-based API.
    "BVH",
    "BatchedBVH",
    "RaggedBVH",
    # Stable foundation classes.
    "BVHHandle",
    "BVHQuery",
    "BatchedBVHHandle",
    "BatchedBVHQuery",
    "RaggedBVHHandle",
    "FPSResult",
    # Stable foundation functions.
    "build_bvh",
    "build_bvh_batched",
    "build_bvh_ragged",
    "bvh_mls_interpolate",
    "bvh_mls_interpolate_batched",
    "mls_interpolate",
    "destroy_bvh",
    "fps",
    "gather_neighbor_values",
    "interpolate_displaced",
    "query_displaced_knn",
    "query_knn",
    "query_knn_batched",
    "query_knn_ragged",
    # Test/debug arithmetic wrappers.
    "implicit_tree_ancestor",
    "implicit_tree_descendant",
    "implicit_tree_summary",
    "morton_encode_2d",
    "morton_encode_3d",
    "morton_split2",
    "morton_split3",
    "smoke_add_one",
    # Public constants.
    "SUPPORTED_K",
    "SUPPORTED_DIMS",
]
