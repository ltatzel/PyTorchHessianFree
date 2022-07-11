"""This script runs the `HessianFree` optimizer on a small test problem."""

import torch

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_OPT = "ggn"  # "hessian" or "ggn"
DAMPING = 1e-3


def get_test_problem():
    """Set-up test problem. Return the model, data and loss function."""

    N = 16
    D_out = 3
    D_hidden = 5
    D_in = 7

    X = torch.rand(N, D_in)
    y = torch.rand(N, D_out)
    data = (X, y)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_hidden),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Linear(D_hidden, D_hidden),
            torch.nn.ReLU(),
        ),
        torch.nn.Linear(D_hidden, D_out),
    )

    # Freeze parameters of first layer --> some parameters not trainable
    first_layer = next(model.children())
    for param in first_layer.parameters():
        param.requires_grad = False

    loss_function = torch.nn.MSELoss(reduction="mean")

    return model, data, loss_function


if __name__ == "__main__":

    torch.manual_seed(0)

    print(f"\nRunning on DEVICE = {DEVICE}")

    # Get problem
    model, data, loss_function = get_test_problem()
    model.to(DEVICE)
    inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)

    def eval_loss_and_outputs():
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        return loss, outputs

    opt = HessianFree(model.parameters(), verbose=True, lr=None)
    for step_idx in range(2):
        print(f"\n========== STEP {step_idx} ==========")
        opt.step(eval_loss_and_outputs)
