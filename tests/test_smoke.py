import torch

import torchbvh


def test_smoke_add_one_cuda():
    assert torch.cuda.is_available()
    x = torch.arange(8, device="cuda", dtype=torch.float32)

    y = torchbvh.smoke_add_one(x)

    torch.testing.assert_close(y, x + 1)
