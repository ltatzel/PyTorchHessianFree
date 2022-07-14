"""This file contains some auxiliary functions."""

import torch


def get_linear_system(dim, seed=0, device="cpu"):
    """Create a random linear system `A x = b` of dimension `dim` with s.p.d.
    system matrix `A`.
    """
    torch.manual_seed(seed)

    A = torch.rand((dim, dim)) - 0.5
    A = A @ A.T + 1e-3 * torch.eye(dim)
    x = torch.rand(dim) - 0.5
    b = A @ x
    return A.to(device), b.to(device), x.to(device)


def get_small_nn_testproblem(N=16, freeze_first_layer=False, device="cpu"):
    """Set-up test problem. Return the model, data and loss function."""

    D_out = 3
    D_hidden = 5
    D_in = 7

    X = torch.rand(N, D_in)
    y = torch.rand(N, D_out)

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
    if freeze_first_layer:
        first_layer = next(model.children())
        for param in first_layer.parameters():
            param.requires_grad = False

    loss_function = torch.nn.MSELoss(reduction="mean")

    return model.to(device), (X.to(device), y.to(device)), loss_function


class TargetFuncModel:
    """Set up a "model" that holds a target function and some parameters."""

    def __init__(self, target_func, init_params):
        self.target_func = target_func
        self.init_params = init_params
        self.params = init_params.clone().detach().requires_grad_(True)

    def parameters(self):
        return self.params

    def eval_loss(self):
        p = self.params
        return self.target_func(p)
