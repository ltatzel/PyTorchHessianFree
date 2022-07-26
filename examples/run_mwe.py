"""A minimal working example using the `HessianFree` optimizer on a small neural
network and some dummy data.
"""

import torch

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
DIM = 10

if __name__ == "__main__":

    # Set up model, loss-function and optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(DIM, DIM, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(DIM, DIM),
    ).to(DEVICE)
    loss_function = loss_function = torch.nn.MSELoss(reduction="mean")
    opt = HessianFree(model.parameters(), verbose=True)

    # Training
    for step_idx in range(5):

        # Sample dummy data, define the `forward`-function
        inputs = torch.rand(BATCH_SIZE, DIM).to(DEVICE)
        targets = torch.rand(BATCH_SIZE, DIM).to(DEVICE)

        def forward():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            return loss, outputs

        # Update the model's parameters
        opt.step(forward=forward)
