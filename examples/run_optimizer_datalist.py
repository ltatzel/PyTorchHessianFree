"""This script runs the `HessianFree` optimizer on a small test problem using
the `step_datalists` method.
"""

import torch
from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_OPT = "ggn"  # "hessian" or "ggn"
DAMPING = 1e-3


# Data
def generate_datalist(N_list):
    datalist = []
    for N in N_list:
        inputs = torch.rand(N, D_in).to(DEVICE)
        targets = torch.rand(N, D_out).to(DEVICE)
        datalist.append((inputs, targets))
    return datalist


if __name__ == "__main__":

    torch.manual_seed(0)

    print(f"\nRunning on DEVICE = {DEVICE}")

    # Problem parameters
    D_in = 7
    D_hidden = 5
    D_out = 3

    forward_datalist = generate_datalist(N_list=[2, 3])
    grad_datalist = generate_datalist(N_list=[3, 4, 1])
    mvp_datalist = generate_datalist(N_list=[3, 1, 5])

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_hidden),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Linear(D_hidden, D_hidden),
            torch.nn.ReLU(),
        ),
        torch.nn.Linear(D_hidden, D_out),
    ).to(DEVICE)

    # Freeze parameters of first layer --> some parameters not trainable
    first_layer = next(model.children())
    for param in first_layer.parameters():
        param.requires_grad = False

    # Loss function
    loss_function = torch.nn.MSELoss(reduction="mean")

    # Set up the optimizer
    opt = HessianFree(model.parameters(), verbose=True)

    # Run the optimizer for a few steps (with default hyperparameters)
    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")
        opt.acc_step(
            model,
            loss_function,
            forward_datalist,
            grad_datalist,  # or `None`
            mvp_datalist,  # or `None`
        )
