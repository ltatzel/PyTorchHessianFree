"""This script uses the `HessianFree` optimizer for training a residual network
on the MNIST data set.
"""

import torch
from example_utils import get_resnet18_mnist_testproblem

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print(f"\nRunning example on DEVICE = {DEVICE}")

    torch.manual_seed(0)

    # Set up problem and optimizer
    model, train_loader, loss_function = get_resnet18_mnist_testproblem(
        batch_size=32, device=DEVICE
    )
    opt = HessianFree(model.parameters(), verbose=True)

    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")

        # Get next mini-batch of data, define `forward` function
        inputs, targets = next(train_loader)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        def forward():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            return loss, outputs

        opt.step(
            forward=forward,
            test_deterministic=True if step_idx == 0 else False,
        )
