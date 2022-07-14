"""Test the preconditioners."""

import pytest
import torch
from hessianfree.preconditioners import (
    sum_grad_squared_autograd,
    sum_grad_squared_backpack,
)

from test_utils import get_small_nn_testproblem

SEEDS = [0, 1, 42]
SEEDS_IDS = [f"seed = {s}" for s in SEEDS]

REDUCTIONS = ["mean", "sum"]
REDUCTIONS_IDS = [f"reduction = {r}" for r in REDUCTIONS]

NS = [1, 16]
NS_IDS = [f"batch size N = {n}" for n in NS]

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
DEVICES_IDS = [f"device = {d}" for d in DEVICES]


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTIONS_IDS)
@pytest.mark.parametrize("N", NS, ids=NS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_sum_grad_squared(seed, reduction, N, device):
    """Test the BackPACK implementation and the autograd implementation for
    computing the sum of the squared gradients. We compute both vectors for a
    small neural network and make sure that they are identical.
    """

    msg = f"seed={seed}, reduction={reduction}, N={N}, device={device}"
    print("\n===== TEST `sum_grad_squared` functions =====\n" + msg)

    # Create test problem
    torch.manual_seed(seed)
    model, data, _ = get_small_nn_testproblem(
        N=N, freeze_first_layer=True, device=device
    )
    inputs, targets = data
    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Call both functions
    vec_ag = sum_grad_squared_autograd(
        model, loss_function, inputs, targets, reduction=reduction
    )
    vec_bp = sum_grad_squared_backpack(model, loss_function, inputs, targets)

    # Analyse
    print("\nvec_ag.shape = ", vec_ag.shape)
    print("vec_ag[:10] = ", vec_ag[:10])

    print("\nvec_bp.shape = ", vec_bp.shape)
    print("vec_bp[:10] = ", vec_bp[:10])
    assert torch.allclose(vec_ag, vec_bp), "Resulting vectors differ."


if __name__ == "__main__":
    test_sum_grad_squared(seed=0, reduction="sum", N=8, device="cpu")
    test_sum_grad_squared(seed=0, reduction="mean", N=8, device="cpu")
