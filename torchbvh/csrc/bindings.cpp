#include <torch/extension.h>

#include "implicit_tree.cuh"
#include "morton.cuh"


torch::Tensor smoke_add_one_cuda(torch::Tensor input);
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int,
    int,
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> build_bvh_cuda(torch::Tensor points);
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> build_bvh_batched_cuda(torch::Tensor points);
std::tuple<torch::Tensor, torch::Tensor> query_knn_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
);
std::tuple<torch::Tensor, torch::Tensor> query_knn_ordered_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
);
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
);
std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_ordered_cuda(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mls_fused_forward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    double regularization,
    double bandwidth_min,
    double exact_eps
);
std::tuple<torch::Tensor, torch::Tensor> morton_sort_queries_batched_cuda(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_exact_full_scan_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    int M
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_metadata_cuda(
    torch::Tensor points,
    torch::Tensor fps_idx,
    torch::Tensor nearest_anchor,
    torch::Tensor nearest_dist_sq,
    torch::Tensor sorted_indices
);
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_approx_bucketed_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    int refresh_interval,
    int candidates_per_round,
    int anchors_per_round,
    double alpha,
    bool use_graph
);
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_exact_bucketed_cuda(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    bool use_graph,
    bool enable_pruning
);
std::tuple<torch::Tensor, torch::Tensor> mls_fused_backward_cuda(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    double bandwidth_min,
    double exact_eps
);


torch::Tensor smoke_add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "smoke_add_one: input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "smoke_add_one: input must be contiguous");
    return smoke_add_one_cuda(input);
}

pybind11::dict build_bvh(torch::Tensor points) {
    auto result = build_bvh_cuda(points);

    pybind11::dict out;
    out["node_aabbs"] = std::get<0>(result);
    out["sorted_indices"] = std::get<1>(result);
    out["scene_min"] = std::get<2>(result);
    out["scene_max"] = std::get<3>(result);
    out["num_leaves"] = std::get<4>(result);
    out["num_real_nodes"] = std::get<5>(result);
    out["leaf_level"] = std::get<6>(result);
    out["virtual_leaves"] = std::get<7>(result);
    out["dim"] = std::get<8>(result);
    out["left_child_mem"] = std::get<9>(result);
    out["right_child_mem"] = std::get<10>(result);
    out["mem_to_leaf"] = std::get<11>(result);
    return out;
}

pybind11::dict build_bvh_batched(torch::Tensor points) {
    auto result = build_bvh_batched_cuda(points);

    pybind11::dict out;
    out["node_aabbs"] = std::get<0>(result);
    out["sorted_indices"] = std::get<1>(result);
    out["scene_min"] = std::get<2>(result);
    out["scene_max"] = std::get<3>(result);
    out["batch_size"] = std::get<4>(result);
    out["num_leaves"] = std::get<5>(result);
    out["num_real_nodes"] = std::get<6>(result);
    out["leaf_level"] = std::get<7>(result);
    out["virtual_leaves"] = std::get<8>(result);
    out["dim"] = std::get<9>(result);
    out["left_child_mem"] = std::get<10>(result);
    out["right_child_mem"] = std::get<11>(result);
    out["mem_to_leaf"] = std::get<12>(result);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> query_knn(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
) {
    return query_knn_cuda(
        node_aabbs,
        sorted_indices,
        query_points,
        num_leaves,
        leaf_level,
        dim,
        k
    );
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_ordered(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int leaf_level,
    int dim,
    int k
) {
    return query_knn_ordered_cuda(
        node_aabbs,
        sorted_indices,
        query_points,
        query_order,
        num_leaves,
        leaf_level,
        dim,
        k
    );
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
) {
    return query_knn_batched_cuda(
        node_aabbs,
        sorted_indices,
        query_points,
        num_leaves,
        num_real_nodes,
        leaf_level,
        dim,
        k
    );
}

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_ordered(
    torch::Tensor node_aabbs,
    torch::Tensor sorted_indices,
    torch::Tensor query_points,
    torch::Tensor query_order,
    int num_leaves,
    int num_real_nodes,
    int leaf_level,
    int dim,
    int k
) {
    return query_knn_batched_ordered_cuda(
        node_aabbs,
        sorted_indices,
        query_points,
        query_order,
        num_leaves,
        num_real_nodes,
        leaf_level,
        dim,
        k
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mls_fused_forward(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    double regularization,
    double bandwidth_min,
    double exact_eps
) {
    return mls_fused_forward_cuda(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        features,
        feature_batch,
        regularization,
        bandwidth_min,
        exact_eps
    );
}

std::tuple<torch::Tensor, torch::Tensor> mls_fused_backward(
    torch::Tensor displaced_points,
    torch::Tensor neighbor_positions,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor features,
    torch::Tensor feature_batch,
    torch::Tensor factors,
    torch::Tensor exact_counts,
    torch::Tensor d_interpolated,
    torch::Tensor d_field_gradient,
    double bandwidth_min,
    double exact_eps
) {
    return mls_fused_backward_cuda(
        displaced_points,
        neighbor_positions,
        indices,
        squared_distances,
        features,
        feature_batch,
        factors,
        exact_counts,
        d_interpolated,
        d_field_gradient,
        bandwidth_min,
        exact_eps
    );
}

std::tuple<torch::Tensor, torch::Tensor> morton_sort_queries_batched(
    torch::Tensor queries,
    torch::Tensor scene_min,
    torch::Tensor scene_max
) {
    return morton_sort_queries_batched_cuda(queries, scene_min, scene_max);
}





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_exact_full_scan(
    torch::Tensor points,
    torch::Tensor seed_indices,
    int M
) {
    return fps_exact_full_scan_cuda(points, seed_indices, M);
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_metadata(
    torch::Tensor points,
    torch::Tensor fps_idx,
    torch::Tensor nearest_anchor,
    torch::Tensor nearest_dist_sq,
    torch::Tensor sorted_indices
) {
    return fps_metadata_cuda(points, fps_idx, nearest_anchor, nearest_dist_sq, sorted_indices);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_approx_bucketed(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    int refresh_interval,
    int candidates_per_round,
    int anchors_per_round,
    double alpha,
    bool use_graph = true
) {
    return fps_approx_bucketed_cuda(
        points,
        seed_indices,
        sorted_indices,
        left_child_mem,
        right_child_mem,
        mem_to_leaf,
        node_aabbs,
        num_real_nodes,
        leaf_level,
        M,
        bucket_size,
        refresh_interval,
        candidates_per_round,
        anchors_per_round,
        alpha,
        use_graph
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> fps_exact_bucketed(
    torch::Tensor points,
    torch::Tensor seed_indices,
    torch::Tensor sorted_indices,
    torch::Tensor left_child_mem,
    torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf,
    torch::Tensor node_aabbs,
    int num_real_nodes,
    int leaf_level,
    int M,
    int bucket_size,
    bool use_graph = true,
    bool enable_pruning = true
) {
    return fps_exact_bucketed_cuda(
        points,
        seed_indices,
        sorted_indices,
        left_child_mem,
        right_child_mem,
        mem_to_leaf,
        node_aabbs,
        num_real_nodes,
        leaf_level,
        M,
        bucket_size,
        use_graph,
        enable_pruning
    );
}




pybind11::dict implicit_tree_summary(int t) {
    TORCH_CHECK(t >= 1, "implicit_tree_summary: t must be >= 1");

    namespace it = implicit_bvh::tree;
    const int leaf = it::leaf_level(t);
    const int max_implicit_nodes = (1 << (leaf + 1)) - 1;

    pybind11::dict out;
    out["t"] = t;
    out["virtual_leaves"] = it::virtual_leaves(t);
    out["real_node_count"] = it::real_node_count(t);
    out["leaf_level"] = leaf;

    pybind11::list virtual_nodes_by_level;
    pybind11::list level_ranges;
    for (int level = 0; level <= leaf; ++level) {
        virtual_nodes_by_level.append(it::virtual_nodes_at_level(t, level));
        const it::Range range = it::level_real_range(t, level);
        pybind11::tuple pair(2);
        pair[0] = range.first;
        pair[1] = range.last;
        level_ranges.append(pair);
    }
    out["virtual_nodes_at_level"] = virtual_nodes_by_level;
    out["level_real_range"] = level_ranges;

    pybind11::list nodes;
    for (int implicit_idx = 0; implicit_idx < max_implicit_nodes; ++implicit_idx) {
        pybind11::dict node;
        node["implicit_idx"] = implicit_idx;
        node["level"] = it::level_of(implicit_idx);
        node["is_virtual"] = it::is_virtual(t, implicit_idx);
        if (!it::is_virtual(t, implicit_idx)) {
            node["memory_index"] = it::memory_index(t, implicit_idx);
        } else {
            node["memory_index"] = pybind11::none();
        }
        node["left_child"] = it::left_child(implicit_idx);
        node["right_child"] = it::right_child(implicit_idx);
        if (implicit_idx == 0) {
            node["parent"] = pybind11::none();
        } else {
            node["parent"] = it::parent(implicit_idx);
        }
        nodes.append(node);
    }
    out["nodes"] = nodes;

    return out;
}

int implicit_tree_descendant(int implicit_idx, int levels_down, int offset) {
    TORCH_CHECK(levels_down >= 0, "implicit_tree_descendant: levels_down must be >= 0");
    TORCH_CHECK(offset >= 0, "implicit_tree_descendant: offset must be >= 0");
    TORCH_CHECK(
        offset < (1 << levels_down),
        "implicit_tree_descendant: offset must be less than 2**levels_down"
    );
    return implicit_bvh::tree::descendant(implicit_idx, levels_down, offset);
}

int implicit_tree_ancestor(int implicit_idx, int levels_up, int descendant_offset) {
    TORCH_CHECK(levels_up >= 0, "implicit_tree_ancestor: levels_up must be >= 0");
    TORCH_CHECK(descendant_offset >= 0, "implicit_tree_ancestor: descendant_offset must be >= 0");
    TORCH_CHECK(
        descendant_offset < (1 << levels_up),
        "implicit_tree_ancestor: descendant_offset must be less than 2**levels_up"
    );
    return implicit_bvh::tree::ancestor(implicit_idx, levels_up, descendant_offset);
}

uint32_t morton_split2(uint32_t x) {
    return implicit_bvh::morton::morton_split2(x);
}

uint32_t morton_split3(uint32_t x) {
    return implicit_bvh::morton::morton_split3(x);
}

uint32_t morton_encode_2d(
    float x,
    float y,
    float min_x,
    float min_y,
    float max_x,
    float max_y
) {
    const float2 scene_min = make_float2(min_x, min_y);
    const float2 scene_max = make_float2(max_x, max_y);
    return implicit_bvh::morton::morton_encode_2d(x, y, scene_min, scene_max);
}

uint32_t morton_encode_3d(
    float x,
    float y,
    float z,
    float min_x,
    float min_y,
    float min_z,
    float max_x,
    float max_y,
    float max_z
) {
    const float3 scene_min = make_float3(min_x, min_y, min_z);
    const float3 scene_max = make_float3(max_x, max_y, max_z);
    return implicit_bvh::morton::morton_encode_3d(x, y, z, scene_min, scene_max);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoke_add_one", &smoke_add_one, "Milestone 1 CUDA smoke test");
    m.def("build_bvh", &build_bvh, "Milestone 4 minimal BVH build");
    m.def("query_knn", &query_knn, "Milestone 6 exact BVH k-NN query");
    m.def("query_knn_ordered", &query_knn_ordered, "Exact BVH k-NN query using sorted query order");
    m.def("build_bvh_batched", &build_bvh_batched, "Milestone 13 fixed-size batched BVH build");
    m.def("query_knn_batched", &query_knn_batched, "Milestone 13 fixed-size batched exact BVH k-NN query");
    m.def("query_knn_batched_ordered", &query_knn_batched_ordered, "Fixed-size batched exact BVH k-NN query using sorted query order");
    m.def("mls_fused_forward", &mls_fused_forward, "Stage 11 private fused MLS forward candidate");
    m.def("mls_fused_backward", &mls_fused_backward, "Stage 11 private fused MLS backward (Submilestone 7E.1)");
    m.def("implicit_tree_summary", &implicit_tree_summary, "Milestone 2 implicit tree arithmetic summary");
    m.def("implicit_tree_descendant", &implicit_tree_descendant, "Implicit tree descendant helper");
    m.def("implicit_tree_ancestor", &implicit_tree_ancestor, "Implicit tree ancestor helper");
    m.def("morton_split2", &morton_split2, "Milestone 3 2D Morton split helper");
    m.def("morton_split3", &morton_split3, "Milestone 3 3D Morton split helper");
    m.def("morton_encode_2d", &morton_encode_2d, "Milestone 3 2D Morton encode helper");
    m.def("morton_encode_3d", &morton_encode_3d, "Milestone 3 3D Morton encode helper");
    m.def("morton_sort_queries_batched", &morton_sort_queries_batched, "Candidate G fused Morton sort for batched query reordering");
    m.def("fps_exact_full_scan", &fps_exact_full_scan, "Exact full-scan FPS fallback");
    m.def("fps_metadata", &fps_metadata, "Stage 12 M2.5 private CUDA FPS metadata helper");
    m.def("fps_approx_bucketed", &fps_approx_bucketed,
        "Approximate bucketed FPS helper",
        py::arg("points"),
        py::arg("seed_indices"),
        py::arg("sorted_indices"),
        py::arg("left_child_mem"),
        py::arg("right_child_mem"),
        py::arg("mem_to_leaf"),
        py::arg("node_aabbs"),
        py::arg("num_real_nodes"),
        py::arg("leaf_level"),
        py::arg("M"),
        py::arg("bucket_size"),
        py::arg("refresh_interval"),
        py::arg("candidates_per_round"),
        py::arg("anchors_per_round"),
        py::arg("alpha"),
        py::arg("use_graph") = true);
    m.def("fps_exact_bucketed", &fps_exact_bucketed,
        "Exact bucketed FPS helper",
        py::arg("points"),
        py::arg("seed_indices"),
        py::arg("sorted_indices"),
        py::arg("left_child_mem"),
        py::arg("right_child_mem"),
        py::arg("mem_to_leaf"),
        py::arg("node_aabbs"),
        py::arg("num_real_nodes"),
        py::arg("leaf_level"),
        py::arg("M"),
        py::arg("bucket_size") = 256,
        py::arg("use_graph") = true,
        py::arg("enable_pruning") = true);
}
