"""Test the accumulation functionality used by `HessianFree`'s `acc_step`-
method.
"""

import torch
from hessianfree.optimizer import HessianFree
import pytest
from test_utils import get_small_nn_testproblem


def generate_datalist(N_list, device="cpu"):
    """Auxiliary function for generating the data list with mini-batch sizes
    given by `N_list`.
    """
    datalist = []
    for N in N_list:
        _, data, _ = get_small_nn_testproblem(N=N, device=device)
        datalist.append(data)
    return datalist


SEEDS = [0, 1, 42]
SEEDS_IDS = [f"seed = {s}" for s in SEEDS]

CURV_OPTS = ["hessian", "ggn"]
CURV_OPTS_IDS = [f"curvature_opt = {c}" for c in CURV_OPTS]

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
DEVICES_IDS = [f"device = {d}" for d in DEVICES]

REDUCTIONS = ["mean", "sum"]
REDUCTIONS_IDS = [f"reduction = {r}" for r in REDUCTIONS]

NUM_MVPS = 5  # Number of matrix-vector products (with random vector)


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("curvature_opt", CURV_OPTS, ids=CURV_OPTS_IDS)
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTIONS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_test_reduction(seed, curvature_opt, reduction, device):
    """Test for `HessianFree`'s `test_reduction` method. With the correct
    reduction (the same reduction that the loss-function uses), the test should
    run without throwing an error. If the wrong reduction is used,
    `test_reduction` should raise an error.
    """

    msg = f"seed={seed}, curvature_opt={curvature_opt} "
    msg += f"reduction={reduction}, device={device}"
    print("\n===== TEST `test_reduction`-method =====\n" + msg)

    # Create test problem
    torch.manual_seed(seed)
    model, _, _ = get_small_nn_testproblem(
        freeze_first_layer=True, device=device
    )
    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Create data lists
    datalist = generate_datalist([4, 3, 7], device)

    # Set up optimizer
    opt = HessianFree(model.parameters(), curvature_opt=curvature_opt)

    # Test with correct reduction
    opt.test_reduction(model, loss_function, datalist, reduction)

    # Test with wrong reduction has to raise an error
    with pytest.raises(Exception):
        wrong_reduction = "mean" if reduction == "sum" else "sum"
        opt.test_reduction(model, loss_function, datalist, wrong_reduction)


if __name__ == "__main__":

    reduction = "sum"

    test_test_reduction(
        seed=0, curvature_opt="ggn", reduction=reduction, device="cpu"
    )
    test_test_reduction(
        seed=0, curvature_opt="hessian", reduction=reduction, device="cpu"
    )
