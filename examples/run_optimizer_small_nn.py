"""In this example, we train a small neural network on some dummy data using the
`HessianFree` optimizer.
"""

import torch
from hessianfree.optimizer import HessianFree

from example_utils import get_small_nn_testproblem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print(f"\nRunning example on DEVICE = {DEVICE}")

    torch.manual_seed(0)

    # Set up problem and optimizer
    model, _, loss_function = get_small_nn_testproblem(device=DEVICE)
    opt = HessianFree(model.parameters(), verbose=True)

    for step_idx in range(2):
        print(f"\n===== STEP {step_idx} =====")

        # Sample new dummy data, define `forward` function
        _, (inputs, targets), _ = get_small_nn_testproblem(device=DEVICE)

        def forward():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            return loss, outputs

        opt.step(
            forward=forward,
            test_deterministic=True if step_idx == 0 else False,
        )
