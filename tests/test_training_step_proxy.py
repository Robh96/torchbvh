import pytest

from tools.training_step_proxy import (
    _fps_policy,
    _geometry_lifecycle_policy,
    _precision_policy,
    _query_count_config,
    build_arg_parser,
    config_from_args,
)


def test_training_step_proxy_parser_defaults_are_smoke_sized():
    args = build_arg_parser().parse_args([])
    config = config_from_args(args)

    assert config.batch_size == 2
    assert config.num_points == 32
    assert config.dim == 3
    assert config.heads == 2
    assert config.channels == 8
    assert config.head_channels == 4
    assert config.k == 4
    assert config.case_name == "proxy_smoke"
    assert config.json_summary is None


def test_training_step_proxy_stage8_metadata_helpers():
    args = build_arg_parser().parse_args(
        ["--B", "3", "--N", "16", "--H", "5", "--case-name", "proxy_smoke_fps"]
    )
    config = config_from_args(args)

    query_counts = _query_count_config(config)
    lifecycle = _geometry_lifecycle_policy(config)
    fps_policy = _fps_policy(config)

    assert query_counts["requested_queries_per_sample"] == 16
    assert query_counts["effective_queries_per_sample"] == 80
    assert query_counts["effective_total_queries"] == 240
    assert query_counts["query_count_source"] == "proxy_flattened_n_times_heads"
    assert _precision_policy()["coordinate_dtype"] == "float32"
    assert lifecycle["fps_construction"] == "once_per_proxy_step"
    assert fps_policy["fps_constructed"] is True
    assert fps_policy["target_tokens"] == 8
    assert config.case_name == "proxy_smoke_fps"


def test_training_step_proxy_rejects_too_few_points_for_k():
    args = build_arg_parser().parse_args(["--N", "4", "--k", "8"])

    with pytest.raises(ValueError, match="--N must be >= --k"):
        config_from_args(args)

