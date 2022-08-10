"""In this example, we train a small neural network on some dummy data using the
`HessianFree` optimizer. Here, we can use different data for the loss, the
gradient and the matrix-vector products.
"""

import torch
from example_utils import get_small_nn_testproblem

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data matching the testproblem from `get_small_nn_testproblem`
def generate_datalist(N_list):
    datalist = []
    for N in N_list:
        _, data, _ = get_small_nn_testproblem(device=DEVICE)
        datalist.append(data)
    return datalist


if __name__ == "__main__":

    print(f"\nRunning example on DEVICE = {DEVICE}")

    torch.manual_seed(0)

    # Set up problem and optimizer
    model, _, loss_function = get_small_nn_testproblem(device=DEVICE)
    opt = HessianFree(model.parameters(), curvature_opt="ggn", verbose=True)

    # Optinal: Test reduction
    test_datalist = generate_datalist([2, 3])
    opt.test_reduction(model, loss_function, test_datalist, reduction="mean")

    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")

        # Sample new dummy data
        loss_datalist = generate_datalist([2, 3])
        grad_datalist = generate_datalist([3, 4, 1])
        mvp_datalist = generate_datalist([3, 1, 5])

        opt.acc_step(
            model,
            loss_function,
            loss_datalist,
            grad_datalist,
            mvp_datalist,
            test_deterministic=True if step_idx == 0 else False,
        )
