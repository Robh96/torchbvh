import pytest

from benchmarks import benchmark_knn


def test_parse_args_defaults_cover_supported_dims_and_k_values():
    args = benchmark_knn.parse_args([])

    assert args.dim == [2, 3]
    assert args.k == [4, 8, 16]
    assert args.n == 10_000
    assert args.queries == 10_000


def test_parse_args_can_narrow_to_one_smoke_case():
    args = benchmark_knn.parse_args(
        [
            "--n",
            "64",
            "--queries",
            "17",
            "--dim",
            "2",
            "--k",
            "4",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--case-name",
            "benchmark_smoke_single",
            "--comparison-label",
            "cpu_ckdtree",
        ]
    )

    assert args.n == 64
    assert args.queries == 17
    assert args.dim == [2]
    assert args.k == [4]
    assert args.iters == 1
    assert args.warmup == 0
    assert args.case_name == "benchmark_smoke_single"
    assert args.comparison_label == "cpu_ckdtree"


def test_parse_args_accepts_torch_cluster_baseline_flag():
    args = benchmark_knn.parse_args(
        [
            "--n",
            "64",
            "--queries",
            "17",
            "--dim",
            "3",
            "--k",
            "4",
            "--torch-cluster-baseline",
        ]
    )

    assert args.torch_cluster_baseline is True


def test_parse_args_accepts_cupy_knn_baseline_flags():
    args = benchmark_knn.parse_args(
        [
            "--n",
            "64",
            "--queries",
            "17",
            "--dim",
            "3",
            "--k",
            "4",
            "--cupy-knn-baseline",
            "--cupy-knn-leaf-size",
            "16",
            "--cupy-knn-compact",
            "--cupy-knn-shrink-to-fit",
            "--no-cupy-knn-sort-queries",
            "--cupy-knn-sort-mode",
            "both",
            "--no-cpu-baseline",
            "--gpu-sort-queries",
        ]
    )

    assert args.cupy_knn_baseline is True
    assert args.no_cpu_baseline is True
    assert args.gpu_sort_queries is True
    assert args.cupy_knn_sort_mode == "both"
    assert benchmark_knn._cupy_knn_options(args) == {
        "leaf_size": 16,
        "compact": True,
        "shrink_to_fit": True,
        "sort_queries": False,
    }
    assert benchmark_knn._cupy_knn_option_variants(args) == [
        ("sorted", {"leaf_size": 16, "compact": True, "shrink_to_fit": True, "sort_queries": True}),
        ("unsorted", {"leaf_size": 16, "compact": True, "shrink_to_fit": True, "sort_queries": False}),
    ]


def test_parse_args_rejects_torch_cluster_for_unsupported_modes():
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(
            [
                "--ragged-n",
                "16",
                "24",
                "--ragged-queries",
                "5",
                "7",
                "--torch-cluster-baseline",
            ]
        )


def test_parse_args_rejects_cupy_knn_for_unsupported_modes():
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(["--cupy-knn-baseline", "--dim", "2"])
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(
            [
                "--cupy-knn-baseline",
                "--dim",
                "3",
                "--ragged-n",
                "16",
                "24",
                "--ragged-queries",
                "5",
                "7",
            ]
        )
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(["--cupy-knn-baseline", "--dim", "3", "--cupy-knn-leaf-size", "0"])
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(["--cupy-knn-baseline", "--dim", "3", "--cupy-knn-shrink-to-fit"])


def test_parse_args_accepts_displaced_query_smoke_flags():
    args = benchmark_knn.parse_args(
        [
            "--batch-size",
            "2",
            "--n",
            "64",
            "--dim",
            "3",
            "--k",
            "4",
            "--displaced-query",
            "--displaced-query-heads",
            "4",
            "--displaced-query-channels",
            "8",
            "--iters",
            "1",
            "--warmup",
            "0",
        ]
    )

    assert args.displaced_query is True
    assert args.displaced_query_heads == 4
    assert args.displaced_query_channels == 8
    assert args.batch_size == 2


def test_effective_query_config_distinguishes_displaced_query_queries():
    args = benchmark_knn.parse_args(
        [
            "--batch-size",
            "2",
            "--n",
            "64",
            "--queries",
            "17",
            "--dim",
            "3",
            "--k",
            "4",
            "--displaced-query",
            "--displaced-query-heads",
            "4",
            "--displaced-query-channels",
            "8",
        ]
    )

    assert benchmark_knn._effective_query_config(args) == {
        "requested_queries": 17,
        "requested_queries_per_sample": 17,
        "effective_queries_per_sample": 256,
        "effective_total_queries": 512,
        "query_count_source": "displaced_query_flattened_n_times_heads",
        "query_count_note": "--displaced-query measures flattened N * H displaced queries per sample; --queries is retained as CLI metadata.",
    }


def test_parse_args_rejects_displaced_query_without_channels():
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(["--displaced-query"])


def test_parse_args_accepts_ragged_and_distribution_flags():
    args = benchmark_knn.parse_args(
        [
            "--ragged-n",
            "16",
            "24",
            "--ragged-queries",
            "5",
            "7",
            "--dim",
            "3",
            "--k",
            "4",
            "--distribution",
            "clustered",
            "--iters",
            "1",
        ]
    )

    assert args.ragged_n == [16, 24]
    assert args.ragged_queries == [5, 7]
    assert args.distribution == "clustered"


def test_effective_query_config_distinguishes_ragged_queries():
    args = benchmark_knn.parse_args(
        [
            "--ragged-n",
            "16",
            "24",
            "--ragged-queries",
            "5",
            "7",
            "--dim",
            "3",
            "--k",
            "4",
            "--queries",
            "99",
        ]
    )

    assert benchmark_knn._effective_query_config(args) == {
        "requested_queries": 99,
        "requested_queries_per_sample": 99,
        "effective_queries_per_sample": [5, 7],
        "effective_total_queries": 12,
        "query_count_source": "ragged_queries",
        "query_count_note": "--ragged-queries controls the measured packed query counts; --queries is retained as CLI metadata.",
    }


def test_parse_args_rejects_too_few_source_points():
    with pytest.raises(SystemExit):
        benchmark_knn.parse_args(["--n", "7", "--k", "8"])


def test_displaced_query_value_gather_shape_and_head_specific_values():
    torch = pytest.importorskip("torch")
    values = torch.arange(1 * 5 * 2 * 3, dtype=torch.float32).reshape(1, 5, 2, 3)
    indices = torch.tensor([[[0, 2], [4, 1], [3, 3], [2, 0], [1, 4], [0, 3], [2, 2], [4, 0], [1, 1], [3, 4]]])

    gathered = benchmark_knn._gather_displaced_query_values(torch, values, indices)

    assert gathered.shape == (1, 5, 2, 2, 3)
    for point in range(5):
        for head in range(2):
            expected = values[0, indices[0, point * 2 + head], head]
            torch.testing.assert_close(gathered[0, point, head], expected)


def test_torch_cluster_dense_conversion_returns_local_batched_indices():
    torch = pytest.importorskip("torch")
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [2, 0, 1, 3, 4, 5, 7, 6],
        ],
        dtype=torch.long,
    )

    dense = benchmark_knn._torch_cluster_dense_conversion(
        torch,
        edge_index,
        batch_size=2,
        queries_per_sample=2,
        n=4,
        k=2,
    )

    assert dense.tolist() == [
        [[2, 0], [1, 3]],
        [[0, 1], [3, 2]],
    ]


def test_query_and_mls_tensor_footprint_estimates_use_clear_families():
    query_footprint = benchmark_knn._query_output_footprint(
        batch_size=2,
        queries_per_sample=5,
        k=4,
        dim=3,
        include_source_position_gather=True,
    )
    mls_footprint = benchmark_knn._mls_tensor_footprint(
        batch_size=2,
        queries_per_sample=5,
        k=4,
        dim=3,
        channels=7,
    )

    assert query_footprint["indices_bytes"] == 2 * 5 * 4 * 8
    assert query_footprint["squared_distances_bytes"] == 2 * 5 * 4 * 4
    assert query_footprint["source_position_gather_bytes"] == 2 * 5 * 4 * 3 * 4
    assert query_footprint["total_bytes"] == sum(
        value for key, value in query_footprint.items() if key != "total_bytes"
    )
    assert mls_footprint["neighbor_feature_gather_bytes"] == 2 * 5 * 4 * 7 * 4
    assert mls_footprint["interpolated_output_bytes"] == 2 * 5 * 7 * 4
    assert mls_footprint["field_gradient_bytes"] == 2 * 5 * 3 * 7 * 4
    assert mls_footprint["total_bytes"] == sum(
        value for key, value in mls_footprint.items() if key != "total_bytes"
    )


def test_displaced_query_tensor_footprint_estimates_outputs_and_value_gather():
    footprint = benchmark_knn._displaced_query_tensor_footprint(
        batch_size=2,
        n=5,
        heads=3,
        k=4,
        dim=3,
        channels=6,
    )

    neighbors = 2 * 5 * 3 * 4
    assert footprint["query_output_indices_bytes"] == neighbors * 8
    assert footprint["query_output_squared_distances_bytes"] == neighbors * 4
    assert footprint["source_position_gather_bytes"] == neighbors * 3 * 4
    assert footprint["value_gather_bytes"] == neighbors * 6 * 4
    assert footprint["total_bytes"] == sum(
        value for key, value in footprint.items() if key != "total_bytes"
    )


def test_cupy_knn_sample_shape_metadata_single_and_batched():
    torch = pytest.importorskip("torch")
    single = benchmark_knn._cupy_knn_sample_shape(
        torch.empty((8, 3), dtype=torch.float32),
        torch.empty((5, 3), dtype=torch.float32),
    )
    batched = benchmark_knn._cupy_knn_sample_shape(
        torch.empty((2, 8, 3), dtype=torch.float32),
        torch.empty((2, 5, 3), dtype=torch.float32),
    )

    assert single == {
        "batch_mode": "single",
        "batch_size": 1,
        "points_shape": [8, 3],
        "query_shape": [5, 3],
        "queries_per_sample": 5,
    }
    assert batched == {
        "batch_mode": "per_sample_loop",
        "batch_size": 2,
        "points_shape": [2, 8, 3],
        "query_shape": [2, 5, 3],
        "queries_per_sample": 5,
    }


def test_cupy_knn_timing_summary_and_speedup_keys_are_consistent():
    summary = benchmark_knn.summarize_timings(
        {
            "gpu_total_bvh_path_ms": [2.0],
            "cupy_knn_build_ms": [1.0],
            "cupy_knn_prepare_ms": [0.5],
            "cupy_knn_query_ms": [3.0],
            "cupy_knn_total_build_prepare_query_ms": [4.0],
        }
    )
    speedup = summary["cupy_knn_total_build_prepare_query_ms"]["mean_ms"] / summary["gpu_total_bvh_path_ms"]["mean_ms"]

    assert summary["cupy_knn_build_ms"]["mean_ms"] == 1.0
    assert summary["cupy_knn_prepare_ms"]["mean_ms"] == 0.5
    assert summary["cupy_knn_query_ms"]["mean_ms"] == 3.0
    assert summary["cupy_knn_total_build_prepare_query_ms"]["mean_ms"] == 4.0
    assert speedup == 2.0


def test_cupy_knn_speedups_support_single_and_sort_variants():
    summary = benchmark_knn.summarize_timings(
        {
            "gpu_total_bvh_path_ms": [10.0],
            "cupy_knn_total_build_prepare_query_ms": [8.0],
            "cupy_knn_sorted_cupy_knn_total_build_prepare_query_ms": [7.0],
            "cupy_knn_unsorted_cupy_knn_total_build_prepare_query_ms": [12.0],
        }
    )

    assert benchmark_knn._cupy_knn_speedups(summary, "gpu_total_bvh_path_ms") == {
        "cupy_knn_total": 0.8,
        "cupy_knn_sorted_total": 0.7,
        "cupy_knn_unsorted_total": 1.2,
    }


def test_cuda_memory_sample_summary_records_start_end_delta_and_peak_fields():
    samples = [
        {
            "cuda_memory_allocated_start_bytes": 10,
            "cuda_memory_allocated_end_bytes": 14,
            "cuda_memory_allocated_delta_bytes": 4,
            "cuda_memory_allocated_peak_bytes": 20,
            "cuda_memory_reserved_start_bytes": 100,
            "cuda_memory_reserved_end_bytes": 120,
            "cuda_memory_reserved_delta_bytes": 20,
            "cuda_memory_reserved_peak_bytes": 128,
        },
        {
            "cuda_memory_allocated_start_bytes": 12,
            "cuda_memory_allocated_end_bytes": 11,
            "cuda_memory_allocated_delta_bytes": -1,
            "cuda_memory_allocated_peak_bytes": 18,
            "cuda_memory_reserved_start_bytes": 120,
            "cuda_memory_reserved_end_bytes": 120,
            "cuda_memory_reserved_delta_bytes": 0,
            "cuda_memory_reserved_peak_bytes": 120,
        },
    ]

    summary = benchmark_knn._summarize_cuda_memory_samples(samples)

    assert summary["measurement_scope"] == "benchmark_case_timed_iteration"
    assert summary["sample_count"] == 2
    assert summary["cuda_memory_allocated_start_bytes_min"] == 10
    assert summary["cuda_memory_allocated_end_bytes_max"] == 14
    assert summary["cuda_memory_allocated_delta_bytes_min"] == -1
    assert summary["cuda_memory_allocated_peak_bytes_max"] == 20
    assert summary["cuda_memory_reserved_peak_bytes_max"] == 128


def test_environment_metadata_includes_benchmark_fields():
    env = benchmark_knn._environment()

    assert env["benchmark_script_version"] == benchmark_knn.BENCHMARK_SCRIPT_VERSION
    assert "benchmark_command" in env
    assert "git" in env
    assert "dirty" in env["git"]
    assert "gpu_total_memory_bytes" in env or "torch_error" in env
    assert "compile_arch" in env


def test_stage8_case_metadata_defaults_and_overrides():
    default_args = benchmark_knn.parse_args(["--n", "64", "--queries", "17", "--dim", "2", "--k", "4"])
    override_args = benchmark_knn.parse_args(
        [
            "--n",
            "64",
            "--queries",
            "17",
            "--dim",
            "2",
            "--k",
            "4",
            "--case-name",
            "local_feasible_single",
            "--comparison-label",
            "custom_baseline",
        ]
    )

    default_metadata = benchmark_knn._stage8_case_metadata(default_args, "single")
    override_metadata = benchmark_knn._stage8_case_metadata(override_args, "single")

    assert default_metadata["case_name"] == "benchmark_single"
    assert default_metadata["result_source"] == "benchmark"
    assert default_metadata["comparison_label"] == "cpu_ckdtree"
    assert default_metadata["comparison_scope"] == "timing_only"
    assert default_metadata["precision_policy"]["coordinate_dtype"] == "float32"
    assert default_metadata["geometry_lifecycle"]["bvh_rebuild"] == "per_warmup_and_timed_iteration"
    assert override_metadata["case_name"] == "local_feasible_single"
    assert override_metadata["comparison_label"] == "custom_baseline"


def test_query_count_summary_labels_requested_and_effective_counts():
    displaced_query_case = {
        "queries": 256,
        "requested_queries": 17,
        "requested_queries_per_sample": 17,
        "effective_queries_per_sample": 256,
        "query_count_source": "displaced_query_flattened_n_times_heads",
    }
    fixed_case = {
        "queries": 17,
        "requested_queries": 17,
        "requested_queries_per_sample": 17,
        "effective_queries_per_sample": 17,
        "query_count_source": "queries",
    }

    assert benchmark_knn._query_count_summary(displaced_query_case) == (
        "requested_queries_per_sample=17 effective_queries_per_sample=256 "
        "query_source=displaced_query_flattened_n_times_heads"
    )
    assert benchmark_knn._query_count_summary(fixed_case) == "queries_per_sample=17"


def test_summarize_timings_result_structure():
    summary = benchmark_knn.summarize_timings(
        {
            "gpu_bvh_build_ms": [1.0, 3.0],
            "cpu_kdtree_query_ms": [2.0],
        }
    )

    assert summary["gpu_bvh_build_ms"]["mean_ms"] == 2.0
    assert summary["gpu_bvh_build_ms"]["min_ms"] == 1.0
    assert summary["gpu_bvh_build_ms"]["max_ms"] == 3.0
    assert summary["gpu_bvh_build_ms"]["stdev_ms"] == pytest.approx(2**0.5)
    assert summary["cpu_kdtree_query_ms"] == {
        "mean_ms": 2.0,
        "stdev_ms": None,
        "min_ms": 2.0,
        "max_ms": 2.0,
    }
