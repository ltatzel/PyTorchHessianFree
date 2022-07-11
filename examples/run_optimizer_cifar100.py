"""This script runs the `HessianFree` optimizer on a small test problem."""

import torch
from config.paths import DATA_DIR
from deepobs.config import set_data_dir
from deepobs.pytorch.runners.runner import PTRunner

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_OPT = "ggn"  # "hessian" or "ggn"
DAMPING = 1e-3


def get_test_problem():
    """Set-up test problem. Return the model, data and loss function."""

    set_data_dir(DATA_DIR)

    # Create testproblem
    tproblem = PTRunner.create_testproblem(
        testproblem="cifar100_allcnnc",
        batch_size=128,
        l2_reg=None,
        random_seed=0,
    )

    # Extract model, loss-function and some training data
    model = tproblem.net
    loss_function = tproblem.loss_function(reduction="mean")
    train_loader, _ = tproblem.data._make_train_and_valid_dataloader()
    data = next(iter(train_loader))

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
