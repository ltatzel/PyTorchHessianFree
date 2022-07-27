"""This file contains functions for setting up example problems (models, data,
loss-functions).
"""

import os

import torch
from deepobs.config import set_data_dir
from deepobs.pytorch.runners.runner import PTRunner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

# Create folder for the data, set DeepOBS data directory
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
set_data_dir(DATA_DIR)


def get_small_nn_testproblem(batch_size=32, freeze_layer1=True, device="cpu"):
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
    if freeze_layer1:
        first_layer = next(model.children())
        for param in first_layer.parameters():
            param.requires_grad = False

    # Dummy data
    inputs = torch.rand(batch_size, D_in).to(device)
    targets = torch.rand(batch_size, D_out).to(device)

    # Loss-function
    loss_function = torch.nn.MSELoss(reduction="mean")

    return model, (inputs, targets), loss_function


def get_allcnnc_cifar100_testproblem(seed=0, batch_size=32, device="cpu"):
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


def get_resnet18_mnist_testproblem(batch_size=32, device="cpu"):
    """Set-up test problem: The model (resnet18-network), train loader (MNIST
    image data) and loss-function.
    """

    # Model
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Training data
    train_set = MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_loader = iter(DataLoader(train_set, batch_size, shuffle=True))

    # Loss-function
    loss_function = nn.CrossEntropyLoss()

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
