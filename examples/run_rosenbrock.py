"""Use the `HessianFree` optimizer to optimize the (deterministic) Rosenbrock-
function.
"""

import torch
from example_utils import TargetFuncModel

from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print(f"\nRunning example on DEVICE = {DEVICE}")

    # Set up problem
    init_params = torch.Tensor([-1.0, 3.0]).to(DEVICE)
    opt_params = torch.Tensor([1.0, 1.0])

    def rosenbrock(xy):
        x, y = xy
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    model = TargetFuncModel(rosenbrock, init_params)

    # Define forward function, set up optimizer
    def forward():
        return model.eval_loss(), None

    opt = HessianFree(
        [model.parameters()], curvature_opt="hessian", adapt_damping=True
    )

    # Optimization
    params_trajectory = [init_params]
    loss_vals = [model.eval_loss().item()]

    for step_idx in range(20):
        print(f"\n===== STEP {step_idx} =====")
        opt.step(forward=forward)

        # Track trajectory and loss
        params_trajectory.append(model.parameters().detach())
        loss_vals.append(model.eval_loss().item())

    # Print trajectory with loss values
    print("\nTrajectory...")
    for (x, y), loss in zip(params_trajectory, loss_vals):
        print(f"x = {x:.3f}, y = {y:.3f}, loss = {loss:.3e}")
    print(f"optimal params: x = {opt_params[0]:.3f}, y = {opt_params[1]:.3f}")
