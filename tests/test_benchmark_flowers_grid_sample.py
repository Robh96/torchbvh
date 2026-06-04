import pytest

from benchmarks import benchmark_flowers_grid_sample as bench


def test_parse_args_preserves_block_defaults():
    args = bench.parse_args([])

    assert args.mode == "block"
    assert args.D == 3
    assert args.grid_resolution == [16, 16, 16]
    assert args.N == 4096
    assert args.H == 8
    assert args.C_effective == 160
    assert args.groups == 8


def test_parse_args_accepts_block_smoke_shape():
    args = bench.parse_args(
        [
            "--grid-resolution",
            "8",
            "8",
            "8",
            "--B",
            "1",
            "--H",
            "2",
            "--C-effective",
            "16",
            "--k",
            "4",
            "--warmup",
            "0",
            "--iters",
            "1",
        ]
    )

    assert args.N == 512
    assert args.groups == 2


def test_parse_args_accepts_mls_fused_diagnostic_flag():
    args = bench.parse_args(["--include-mls-fused"])

    assert args.include_mls_fused is True


def test_parse_args_rejects_groupnorm_mismatch_for_block():
    with pytest.raises(SystemExit):
        bench.parse_args(["--C-effective", "16", "--H", "2", "--groups", "3"])


def test_result_common_labels_packed_public_geometry_lifecycle():
    args = bench.parse_args(["--grid-resolution", "8", "8", "8", "--C-effective", "16", "--H", "2"])

    result = bench._result_common(args, "bvh_block_proxy")

    assert result["comparison_tier"] == "end_to_end_block_comparison"
    assert result["comparison_label"] == "bvh_block_proxy"
    assert result["diagnostic_for"] is None
    assert result["pipeline_role"] == "public_bvh_proxy"
    assert result["effective_queries_per_sample"] == 1024
    assert result["geometry_lifecycle"]["bvh_rebuild"] == "package_internal_fused_flattened_head_build_bvh_batched_once_per_forward"


def test_result_common_labels_exact_fused_diagnostic_geometry_lifecycle():
    args = bench.parse_args(["--include-mls-fused"])

    result = bench._result_common(args, bench.DIAGNOSTIC_FUSED_LABEL)

    assert result["comparison_tier"] == "diagnostic_breakdown_not_comparable"
    assert result["diagnostic_for"] == "bvh_block_proxy"
    assert result["pipeline_role"] == "instrumentation_only"
    assert result["geometry_lifecycle"]["bvh_rebuild"] == "benchmark_local_exact_fused_flattened_head_build_bvh_batched_once_per_forward"
    assert result["geometry_lifecycle"]["handle_destroy"] == "benchmark_local_after_query_before_fused_mls"


def test_result_common_labels_split_query_diagnostic_as_not_comparable():
    args = bench.parse_args(["--include-mls-fused", "--reorder-queries"])

    result = bench._result_common(args, bench.DIAGNOSTIC_SPLIT_QUERY_LABEL)

    assert result["comparison_tier"] == "diagnostic_breakdown_not_comparable"
    assert result["diagnostic_for"] == "bvh_block_proxy"
    assert result["geometry_lifecycle"]["query_reorder"] is True
