"""Test the preconditioners."""

import pytest
import torch
from test_utils import get_small_nn_testproblem
from torch.nn.utils.convert_parameters import parameters_to_vector

from hessianfree.preconditioners import (
    diag_EF_autograd,
    diag_EF_backpack,
    diag_to_preconditioner,
)


def empirical_fisher_autograd(
    model, loss_function, inputs, targets, reduction, device
):
    """Compute the empirical Fisher matrix using autograd."""

    params_list = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params_list)

    # Compute individual gradients, add outer product to `F`
    F = torch.zeros((num_params, num_params)).to(device)
    for (input_i, target_i) in zip(inputs, targets):
        loss_i = loss_function(model(input_i), target_i)
        grad_i = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        grad_i = parameters_to_vector(grad_i)
        F = F + torch.outer(grad_i, grad_i)

    # Fix scaling for reduction `"mean"`
    if reduction == "mean":
        N = inputs.shape[0]
        F = F / N

    return F


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
def test_diag_EF(seed, reduction, N, device):
    """Test the BackPACK implementation and the autograd implementation for
    computing the diagonal of the empirical Fisher matrix. We compute both
    vectors for a small neural network and make sure that they are identical.
    Additionally, we compute the entire empirical Fisher matrix using the
    auxiliary function above, extract its diagonal and compare to the other
    implementations.
    """

    msg = f"seed={seed}, reduction={reduction}, N={N}, device={device}"
    print("\n===== TEST `diag_EF` functions =====\n" + msg)

    # Create test problem
    torch.manual_seed(seed)
    model, data, _ = get_small_nn_testproblem(
        N=N, freeze_first_layer=True, device=device
    )
    inputs, targets = data
    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Call both functions
    diag_EF_ag = diag_EF_autograd(
        model, loss_function, inputs, targets, reduction=reduction
    )
    diag_EF_bp = diag_EF_backpack(
        model, loss_function, inputs, targets, reduction=reduction
    )

    # Compute the empirical Fisher matrix, extract the diagonal
    diag_EF = torch.diag(
        empirical_fisher_autograd(
            model, loss_function, inputs, targets, reduction, device
        )
    )

    # Analyse
    print("diag_EF_ag[:10] = ", diag_EF_ag[:10])
    print("diag_EF_bp[:10] = ", diag_EF_bp[:10])
    print("diag_EF[:10] = ", diag_EF[:10])
    assert torch.allclose(diag_EF_ag, diag_EF_bp), "Resulting diagonals differ."
    assert torch.allclose(diag_EF_ag, diag_EF), "Resulting diagonals differ."
    assert torch.allclose(diag_EF_bp, diag_EF), "Resulting diagonals differ."


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_diag_to_preconditioner(seed, device):
    """Test the preconditioner recipe: Make sure that `M_func` created by
    `diag_to_preconditioner` concatinated by `P` (see below), is the identity.
    """

    msg = f"seed={seed}, device={device}"
    print("\n===== TEST `diag_to_preconditioner` =====\n" + msg)

    torch.manual_seed(seed)

    # Parameters
    dim = 10
    damping = 0.1
    exponent = 0.75

    # Build preconditioner and `P`
    diag_FE = torch.rand(dim)
    P = torch.diag((diag_FE + damping) ** exponent)
    M_func = diag_to_preconditioner(diag_FE, damping, exponent)

    # The concatenation of `M_func` and `P` should be the identity
    for _ in range(5):
        vec = torch.rand(dim)
        assert torch.allclose(P @ M_func(vec), vec)


if __name__ == "__main__":
    test_diag_EF(seed=0, reduction="sum", N=8, device="cpu")
    test_diag_EF(seed=0, reduction="mean", N=8, device="cpu")
    test_diag_to_preconditioner(seed=0, device="cpu")
