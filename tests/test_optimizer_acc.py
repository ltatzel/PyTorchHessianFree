"""Test the accumulation functionality used by `HessianFree`'s `acc_step`-
method.
"""

import torch
from hessianfree.optimizer import HessianFree
from torch.nn.utils.convert_parameters import parameters_to_vector
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


SEEDS = [0, 1, 42]
SEEDS_IDS = [f"seed = {s}" for s in SEEDS]

CURV_OPTS = ["hessian", "ggn"]
CURV_OPTS_IDS = [f"curvature_opt = {c}" for c in CURV_OPTS]

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
DEVICES_IDS = [f"device = {d}" for d in DEVICES]

REDUCTIONS = ["mean"]  # TODO: Add "sum"
REDUCTIONS_IDS = [f"reduction = {r}" for r in REDUCTIONS]

NUM_MVPS = 5  # Number of matrix-vector products (with random vector)


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("curvature_opt", CURV_OPTS, ids=CURV_OPTS_IDS)
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=REDUCTIONS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_on_neural_network(seed, curvature_opt, reduction, device):
    """TODO"""

    # Create test problem
    torch.manual_seed(seed)
    model, _, _ = get_small_nn_testproblem(
        freeze_first_layer=True, device=device
    )
    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Create data lists
    forward_datalist = generate_datalist([1, 2, 3], device)
    ref_inputs, ref_targets = datalist_to_data(forward_datalist)

    # Set up optimizer
    opt = HessianFree(
        model.parameters(), curvature_opt=curvature_opt, verbose=True
    )

    # --------------------------------------------------------------------------
    # Loss and Outputs
    # --------------------------------------------------------------------------

    # Create forward lists
    losses_list, outputs_list, N_list = opt._forward_lists(
        model, loss_function, forward_datalist
    )

    # Loss and outputs
    ref_outputs = model(ref_inputs)
    ref_loss = loss_function(ref_outputs, ref_targets)
    loss, outputs = opt._acc_loss_and_outputs(losses_list, outputs_list, N_list)
    assert torch.allclose(loss, ref_loss), "Inconsistent loss"
    assert torch.allclose(outputs, ref_outputs), "Inconsistent outputs"

    # --------------------------------------------------------------------------
    # Gradient
    # --------------------------------------------------------------------------

    grad_datalist = forward_datalist

    # Create forward lists
    losses_list, outputs_list, N_list = opt._forward_lists(
        model, loss_function, grad_datalist
    )

    # Gradient
    ref_grad = torch.autograd.grad(ref_loss, opt._params_list)
    ref_grad = parameters_to_vector(ref_grad).detach()
    grad = opt._acc_grad(losses_list, N_list)
    assert torch.allclose(grad, ref_grad), "Inconsistent gradient"

    # --------------------------------------------------------------------------
    # Matrix-vector products
    # --------------------------------------------------------------------------

    mvp_datalist = forward_datalist

    # Create forward lists
    losses_list, outputs_list, N_list = opt._forward_lists(
        model, loss_function, mvp_datalist
    )

    ref_outputs = model(ref_inputs)
    ref_loss = loss_function(ref_outputs, ref_targets)

    for _ in range(NUM_MVPS):

        # Sample random vector
        x = torch.rand(grad.shape)

        # Matrix-vector product with `x`
        if curvature_opt == "ggn":
            ref_mvp = opt._Gv(ref_loss, ref_outputs, opt._params_list, x)
        elif curvature_opt == "hessian":
            ref_mvp = opt._Hv(ref_loss, opt._params_list, x)
        mvp = opt._acc_mvp(losses_list, outputs_list, N_list, x)
        assert torch.allclose(mvp, ref_mvp), "Inconsistent mvp"

    print("All tests passed!")


if __name__ == "__main__":
    test_on_neural_network(
        seed=0, curvature_opt="ggn", reduction="mean", device="cpu"
    )
    test_on_neural_network(
        seed=0, curvature_opt="hessian", reduction="mean", device="cpu"
    )
