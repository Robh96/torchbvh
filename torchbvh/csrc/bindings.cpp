#include <torch/extension.h>

using BvhBuildResult = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int, int, int, int,
    torch::Tensor, torch::Tensor, torch::Tensor>;

BvhBuildResult build_bvh_batched_cooperative_cuda(torch::Tensor points);
BvhBuildResult build_primitive_bvh_batched_cuda(torch::Tensor primitives);

torch::Tensor morton_sort_points_batched_cuda(torch::Tensor points);
torch::Tensor morton_sort_queries_narrow_cuda(
    torch::Tensor queries, torch::Tensor scene_min, torch::Tensor scene_max);
torch::Tensor morton_sort_routed_queries_batched_cuda(
    torch::Tensor queries, torch::Tensor routes,
    torch::Tensor true_scene_min, torch::Tensor true_scene_max,
    torch::Tensor false_scene_min, torch::Tensor false_scene_max);

std::tuple<torch::Tensor, torch::Tensor> query_knn_batched_cached_bounds_cuda(
    torch::Tensor node_aabbs, torch::Tensor sorted_indices,
    torch::Tensor query_points, int num_leaves, int num_real_nodes,
    int dim, int k);
std::tuple<torch::Tensor, torch::Tensor>
query_knn_batched_cached_bounds_ordered_cuda(
    torch::Tensor node_aabbs, torch::Tensor sorted_indices,
    torch::Tensor query_points, torch::Tensor query_order,
    int num_leaves, int num_real_nodes, int dim, int k);
std::tuple<torch::Tensor, torch::Tensor>
query_knn_batched_cached_bounds_spatial_cuda(
    torch::Tensor node_aabbs, torch::Tensor sorted_indices,
    torch::Tensor query_points, torch::Tensor query_order,
    int num_leaves, int num_real_nodes, int dim, int k);
std::tuple<torch::Tensor, torch::Tensor>
query_knn_routed_batched_cached_bounds_spatial_cuda(
    torch::Tensor true_node_aabbs, torch::Tensor true_sorted_indices,
    torch::Tensor false_node_aabbs, torch::Tensor false_sorted_indices,
    torch::Tensor query_points, torch::Tensor routes, torch::Tensor query_order,
    int true_num_leaves, int true_num_real_nodes,
    int false_num_leaves, int false_num_real_nodes, int dim, int k);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mls_packed_indexed_forward_cuda(
    torch::Tensor displaced_points, torch::Tensor source_points,
    torch::Tensor indices, torch::Tensor squared_distances,
    torch::Tensor features, torch::Tensor query_order,
    int queries_per_batch, int queries_per_head,
    double regularization, double bandwidth_min, double exact_eps);
std::tuple<torch::Tensor, torch::Tensor> mls_packed_indexed_backward_cuda(
    torch::Tensor displaced_points, torch::Tensor source_points,
    torch::Tensor indices, torch::Tensor squared_distances,
    torch::Tensor features, torch::Tensor query_order,
    torch::Tensor factors, torch::Tensor exact_counts,
    torch::Tensor d_interpolated, torch::Tensor d_field_gradient,
    int queries_per_batch, int queries_per_head,
    double bandwidth_min, double exact_eps);

std::tuple<torch::Tensor, torch::Tensor> raytrace_batched_cuda(
    torch::Tensor node_aabbs, torch::Tensor sorted_indices,
    torch::Tensor left_child_mem, torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf, torch::Tensor primitives,
    torch::Tensor origins, torch::Tensor directions, int num_real_nodes,
    double t_min, double t_max);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
raytrace_batched_cached_cuda(
    torch::Tensor node_aabbs, torch::Tensor sorted_indices,
    torch::Tensor left_child_mem, torch::Tensor right_child_mem,
    torch::Tensor mem_to_leaf, torch::Tensor primitives,
    torch::Tensor origins, torch::Tensor directions, int num_real_nodes,
    double t_min, double t_max);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
raytrace_segment_backward_cuda(
    torch::Tensor indices, torch::Tensor hit_t, torch::Tensor primitives,
    torch::Tensor origins, torch::Tensor directions, torch::Tensor grad_t,
    torch::Tensor grad_points, bool need_primitives, bool need_origins,
    bool need_directions);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fps_exact_bucketed_lean_cuda(
    torch::Tensor points, torch::Tensor seed_indices,
    torch::Tensor sorted_indices, int M, int bucket_size,
    bool enable_pruning, bool use_graph);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_metadata_cuda(
    torch::Tensor points, torch::Tensor fps_idx,
    torch::Tensor nearest_anchor, torch::Tensor nearest_dist_sq,
    torch::Tensor sorted_indices);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fps_approx_bucketed_lean_cuda(
    torch::Tensor points, torch::Tensor seed_indices,
    torch::Tensor sorted_indices, int M,
    int bucket_size, int refresh_interval, int candidates_per_round,
    int anchors_per_round, double alpha, bool use_graph);

pybind11::dict make_bvh_dict(const BvhBuildResult& result) {
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

pybind11::dict build_bvh_batched_cooperative(torch::Tensor points) {
    return make_bvh_dict(build_bvh_batched_cooperative_cuda(points));
}

pybind11::dict build_primitive_bvh_batched(torch::Tensor primitives) {
    return make_bvh_dict(build_primitive_bvh_batched_cuda(primitives));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bvh_batched_cooperative", &build_bvh_batched_cooperative);
    m.def("build_primitive_bvh_batched", &build_primitive_bvh_batched);
    m.def("morton_sort_points_batched", &morton_sort_points_batched_cuda);
    m.def("morton_sort_queries_narrow", &morton_sort_queries_narrow_cuda);
    m.def("morton_sort_routed_queries_batched", &morton_sort_routed_queries_batched_cuda);
    m.def("query_knn_batched_cached_bounds", &query_knn_batched_cached_bounds_cuda);
    m.def("query_knn_batched_cached_bounds_ordered", &query_knn_batched_cached_bounds_ordered_cuda);
    m.def("query_knn_batched_cached_bounds_spatial", &query_knn_batched_cached_bounds_spatial_cuda);
    m.def("query_knn_routed_batched_cached_bounds_spatial",
          &query_knn_routed_batched_cached_bounds_spatial_cuda);
    m.def("mls_packed_indexed_forward", &mls_packed_indexed_forward_cuda);
    m.def("mls_packed_indexed_backward", &mls_packed_indexed_backward_cuda);
    m.def("raytrace_batched", &raytrace_batched_cuda);
    m.def("raytrace_batched_cached", &raytrace_batched_cached_cuda);
    m.def("raytrace_segment_backward", &raytrace_segment_backward_cuda);
    m.def(
        "fps_exact_bucketed_lean", &fps_exact_bucketed_lean_cuda,
        pybind11::arg("points"), pybind11::arg("seed_indices"),
        pybind11::arg("sorted_indices"), pybind11::arg("M"),
        pybind11::arg("bucket_size"), pybind11::arg("enable_pruning") = true,
        pybind11::arg("use_graph") = true);
    m.def("fps_metadata", &fps_metadata_cuda);
    m.def(
        "fps_approx_bucketed_lean",
        &fps_approx_bucketed_lean_cuda,
        pybind11::arg("points"), pybind11::arg("seed_indices"),
        pybind11::arg("sorted_indices"), pybind11::arg("M"),
        pybind11::arg("bucket_size"), pybind11::arg("refresh_interval"),
        pybind11::arg("candidates_per_round"), pybind11::arg("anchors_per_round"),
        pybind11::arg("alpha"), pybind11::arg("use_graph") = true);

}
