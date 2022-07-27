"""Test the accumulation functionality used by `HessianFree`'s `acc_step`-
method.
"""

from copy import deepcopy

import pytest
import torch
from test_utils import get_small_nn_testproblem

from hessianfree.optimizer import HessianFree


def generate_datalist(N_list, device="cpu"):
    """Auxiliary function for generating the data list with mini-batch sizes
    given by `N_list`.
    """
    datalist = []
    for N in N_list:
        _, data, _ = get_small_nn_testproblem(N=N, device=device)
        datalist.append(data)
    return datalist


def datalist_to_data(datalist):
    """Auxiliary function for converting a data list into an `(inputs,
    targets)`-tuple.
    """
    inputs_list = []
    targets_list = []
    for inputs, targets in datalist:
        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs = torch.cat(inputs_list, dim=0).clone()
    targets = torch.cat(targets_list, dim=0).clone()
    return inputs, targets


def check_models_equal(m1, m2, atol=1e-4):
    """Check if two models are identical."""
    equal = True

    m1_params = m1.parameters()
    m2_params = m2.parameters()

    for p_idx, (p1, p2) in enumerate(zip(m1_params, m2_params)):
        p1_data = p1.data
        p2_data = p2.data
        if not torch.allclose(p1_data, p2_data, atol=atol):
            equal = False
            diff_norm = torch.linalg.norm(p1_data - p2_data)
            print(f"Difference at p_idx = {p_idx}, diff_norm = {diff_norm}")

    if not equal:
        raise RuntimeError("Models are not identical.")
    print("Models are identical.")


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
    model, _, _ = get_small_nn_testproblem(freeze_layer1=True, device=device)
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


N_LISTS = [[16], [7, 8]]
N_LISTS_IDS = [f"N_list = {N_list}" for N_list in N_LISTS]


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("curvature_opt", CURV_OPTS, ids=CURV_OPTS_IDS)
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTIONS_IDS)
@pytest.mark.parametrize("N_list", N_LISTS, ids=N_LISTS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.filterwarnings("ignore:The reduction ratio `rho` is negative.")
@pytest.mark.filterwarnings("ignore:Directional curvature pAp")
@pytest.mark.filterwarnings("ignore:`update_vec`-parameter in")
def test_step(seed, curvature_opt, reduction, N_list, device):
    """Create two identical models and set up two corresponding optimizers: One
    uses the `step`-method on the `(inputs, targets)`-tuple, the other one uses
    the `acc_step`-method on the same data but in form of a data list. We test
    that both trajectories in parameter space are "identical".
    """

    msg = f"seed={seed}, curvature_opt={curvature_opt} "
    msg += f"reduction={reduction}, N_list={N_list}, device={device}"
    print("\n===== TEST step =====\n" + msg)

    # Create test problem with two identical models
    torch.manual_seed(seed)
    model_1, _, _ = get_small_nn_testproblem(freeze_layer1=True, device=device)
    model_2 = deepcopy(model_1)
    check_models_equal(model_1, model_2)

    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Set up optimizers
    opt_1 = HessianFree(
        model_1.parameters(),
        curvature_opt=curvature_opt,
        cg_max_iter=4,
        verbose=True,
    )
    opt_2 = HessianFree(
        model_2.parameters(),
        curvature_opt=curvature_opt,
        cg_max_iter=4,
        verbose=True,
    )

    for step_idx in range(3):

        # Sample and organize data
        datalist = generate_datalist(N_list, device)
        inputs, targets = datalist_to_data(datalist)

        def forward():
            outputs = model_1(inputs)
            loss = loss_function(outputs, targets)
            return loss, outputs

        # Compute step and compare models afterwards
        print(f"\n===== STEP {step_idx} opt_1 =====")
        opt_1.step(forward=forward)

        print(f"\n===== STEP {step_idx} opt_2 =====")
        opt_2.acc_step(model_2, loss_function, datalist, reduction=reduction)

        check_models_equal(model_1, model_2)


if __name__ == "__main__":

    reduction = "sum"

    test_test_reduction(
        seed=0, curvature_opt="ggn", reduction=reduction, device="cpu"
    )
    test_test_reduction(
        seed=0, curvature_opt="hessian", reduction=reduction, device="cpu"
    )

    test_step(
        seed=1,
        curvature_opt="ggn",
        reduction=reduction,
        N_list=[7, 8],
        device="cpu",
    )
