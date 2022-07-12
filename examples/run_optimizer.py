"""This script runs the `HessianFree` optimizer on a small test problem."""

import torch
from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_OPT = "ggn"  # "hessian" or "ggn"
DAMPING = 1e-3


if __name__ == "__main__":

    torch.manual_seed(0)

    print(f"\nRunning on DEVICE = {DEVICE}")

    # Problem parameters
    N = 16  # Batch size
    D_in = 7
    D_hidden = 5
    D_out = 3

    # Data
    inputs = torch.rand(N, D_in).to(DEVICE)
    targets = torch.rand(N, D_out).to(DEVICE)

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
    def eval_loss_and_outputs():
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        return loss, outputs

    opt = HessianFree(model.parameters(), verbose=True)

    # Run the optimizer for a few steps (with default hyperparameters)
    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")
        opt.step(eval_loss_and_outputs)
