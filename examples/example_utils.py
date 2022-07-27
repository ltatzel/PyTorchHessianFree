"""This file contains functions for setting up example problems (models, data,
loss-functions).
"""

import torch
from deepobs.pytorch.runners.runner import PTRunner


def get_small_nn_testproblem(
    batch_size=16,
    freeze_first_layer=True,
    device="cpu",
):
    """Set-up test problem: The model (a small neural network), data and loss-
    function.
    """

    # In- and output dimensions
    D_in = 7
    D_hidden = 5
    D_out = 3

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_hidden),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Linear(D_hidden, D_hidden),
            torch.nn.ReLU(),
        ),
        torch.nn.Linear(D_hidden, D_out),
    ).to(device)

    # Freeze parameters of first layer --> some parameters not trainable
    if freeze_first_layer:
        first_layer = next(model.children())
        for param in first_layer.parameters():
            param.requires_grad = False

    # Dummy data
    inputs = torch.rand(batch_size, D_in).to(device)
    targets = torch.rand(batch_size, D_out).to(device)

    # Loss-function
    loss_function = torch.nn.MSELoss(reduction="mean")

    return model, (inputs, targets), loss_function


def get_cifar100_testproblem(batch_size=32, seed=0, device="cpu"):
    """Set-up test problem: The model (ALLCNNC-network), train loader (CIFAR-100
    image data) and loss-function.
    """

    # Create DeepOBS testproblem
    tproblem = PTRunner.create_testproblem(
        testproblem="cifar100_allcnnc",
        batch_size=batch_size,
        l2_reg=None,
        random_seed=seed,
    )

    # Extract model, loss-function and data
    model = tproblem.net
    loss_func = tproblem.loss_function(reduction="mean")
    train_loader, _ = tproblem.data._make_train_and_valid_dataloader()

    # Regularized loss
    def loss_function(outputs, targets):
        loss = loss_func(outputs, targets)
        l2_loss = tproblem.get_regularization_loss()
        return loss + l2_loss

    return model.to(device), train_loader, loss_function


class TargetFuncModel:
    """Set up a "model" that holds a target function and some parameters."""

    def __init__(self, target_func, init_params):
        self.target_func = target_func
        self.init_params = init_params
        self.params = init_params.clone().detach().requires_grad_(True)

    def parameters(self):
        return self.params

    def eval_loss(self):
        return self.target_func(self.params)
