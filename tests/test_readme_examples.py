"""Keep the copy-paste quickstart in sync with the public CUDA API."""

from pathlib import Path
import re

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_readme_quickstart_runs_end_to_end():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quickstart\n", 1)[1].split("\n## References", 1)[0]
    examples = re.findall(r"```python\n(.*?)\n```", quickstart, re.DOTALL)
    assert len(examples) == 2

    namespace = {}
    for example in examples:
        exec(compile(example, "README.md", "exec"), namespace)

    assert namespace["indices"].shape == (16, 8)
    assert namespace["batch_indices"].shape == (2, 8, 4)
    assert namespace["samples"].indices.shape == (32,)
    assert namespace["sampled"].shape == (2, 8, 2, 4)


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
