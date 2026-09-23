"""Keep the copy-paste quickstart in sync with the public CUDA API."""

from pathlib import Path
import re

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_readme_quickstart_runs_end_to_end():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quickstart\n", 1)[1].split("\n## ", 1)[0]
    examples = re.findall(r"```python\n(.*?)\n```", quickstart, re.DOTALL)
    assert len(examples) == 1

    namespace = {}
    for example in examples:
        exec(compile(example, "README.md", "exec"), namespace)

    assert namespace["neighbors"].shape == (10_000, 4)
    assert namespace["distances_sq"].shape == (10_000, 4)
    assert torch.isfinite(namespace["distances_sq"]).all()
    assert namespace["sample"].indices.shape == (2_500,)
    assert namespace["values"].shape == (10_000, 64)
    assert torch.isfinite(namespace["values"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_user_guide_ray_and_conditional_examples():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "user_guide.md").read_text(
        encoding="utf-8"
    )
    namespace = {"torch": torch}
    import torchbvh as tb

    namespace["tb"] = tb
    for heading in ("Ray Tracing", "Conditional MLS Routing"):
        section = guide.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
        example = re.findall(r"```python\n(.*?)\n```", section, re.DOTALL)[0]
        exec(compile(example, "docs/user_guide.md", "exec"), namespace)

    assert namespace["hits"].mask.tolist() == [True]
    assert namespace["values"].shape == (2, 8, 2, 4)
