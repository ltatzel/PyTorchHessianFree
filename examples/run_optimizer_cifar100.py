"""This script runs the `HessianFree` optimizer on the DeepOBS
`cifar100_allcnnc` test problem."""

import torch
from deepobs.pytorch.runners.runner import PTRunner
from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_OPT = "ggn"  # "hessian" or "ggn"
DAMPING = 1e-3


def get_cifar100_testproblem():
    """Set-up test problem. Return the model, data and loss function."""

    # set_data_dir(DATA_DIR)

    # Create testproblem
    tproblem = PTRunner.create_testproblem(
        testproblem="cifar100_allcnnc",
        batch_size=32,
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

    # Get problem, move to `DEVICE`
    model, data, loss_function = get_cifar100_testproblem()
    model.to(DEVICE)
    inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)

    def eval_loss_and_outputs():
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        return loss, outputs

    opt = HessianFree(model.parameters(), verbose=True)
    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")
        opt.step(eval_loss_and_outputs)
