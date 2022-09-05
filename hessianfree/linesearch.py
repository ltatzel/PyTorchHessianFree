"""Implementation of the line search."""

from warnings import warn

import torch


def simple_linesearch(
    f,
    f_grad_0,
    step,
    init_alpha=1.0,
    beta=0.8,
    c=1e-2,
    max_iter=20,
    verbose=False,
):
    """Iteratively reduce the scale of the update vector `step` by the factor
    `beta`, until the target function `f` is decreased "significantly" (Armijo
    condition). More precisely, `alpha` is chosen as step size if
    `f(alpha * step) <= f(0) + alpha * c * f_grad_0^T * step`. This approach is
    described in [2, Section 8.8].

    Args:
        f (callable): Target function: Maps a step `step` to the corresonding
            value (a float) of the target function.
        f_grad_0 (torch.Tensor): The target function's gradient at `0`. This is
            required for the termination criterion of the line search.
        step (torch.Tensor): The initially proposed update step.
        init_alpha (float): Use `init_alpha` as the first guess for the
            step size. Defaults to `1.0`.
        beta (float): The step size reduction factor: In iteration `i` (starting
            with `i = 0`) of the line search, we try the step size
            `init_alpha * beta^i`.
        c (float): This parameter defines what is considered as
            "significant" improvement. The default value is taken from [2,
            Section 8.8].
        max_iter (int): The maximum number of search steps.
        verbose (bool): Print information during the line search.

    Returns:
        A tuple containing
        alpha (float): The step size.
        f_alpha_step (float): The target function value `f(alpha * step)`.

    Note: If no step size can be found that fulfills the above termination
    criterion, the tuple `(0.0, f_0)` is returned.
    """

    # Check parameters
    if beta >= 1.0:
        raise ValueError(f"Invalid reduction factor beta = {beta}")

    if c < 0.0:
        raise ValueError(f"Invalid c = {c}")

    if verbose:
        print("\nStarting line search...")

    # Value of target function at zero
    f_0 = float(f(torch.zeros_like(step)))
    if verbose:
        print(f"  f(0) = {f_0:.6f}")

    # Value of target function at `init_alpha * step`
    f_init_alpha_step = float(f(init_alpha * step))
    if verbose:
        print(f"  f(init_alpha * step) = {f_init_alpha_step:.6f}")

    # The directional derivative multiplied by c
    c_direc_deriv = c * torch.dot(f_grad_0, step).item()
    if c_direc_deriv >= 0:
        msg = "`update_vec`-parameter in `simple_linesearch` is not a descent "
        msg += f"direction. The directional derivative is {c_direc_deriv:.6f}."
        warn(msg)

    # Initialize
    alpha = init_alpha
    f_alpha_step = f_init_alpha_step

    # Scale `step` until "significant" improvement in `f`
    for _ in range(max_iter):
        if verbose:
            msg = f"  Trying alpha = {alpha:.6f}, "
            msg += f"f(alpha * step) = {f_alpha_step:.6f}"
            print(msg)

        # Check termination condition
        if float(f_alpha_step) <= f_0 + alpha * c_direc_deriv:
            if verbose:
                print(f"Significant improvement for alpha = {alpha:.6f}")
            return alpha, f_alpha_step

        # Reduce scale
        else:
            alpha *= beta
            f_alpha_step = f(alpha * step)

    # No suitable update found
    warn("No suitable update could be found by the line search.")
    if verbose:
        print(f"No significant improvement. Using alpha = {0.0:.6f}")
    return 0.0, f_0
