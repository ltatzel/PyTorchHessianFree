"""Implementation of cg-backtracking."""

import torch


def cg_backtracking(f, steps_list, verbose=False):
    """Evaluate the target function `f` for all update steps in `steps_list`.
    Return the minimum target function value and the corresonding list index.

    Args:
        f (callable): Target function: Maps a step `step` to the corresonding
            value (a float) of the target function.
        steps_list (list): Contains a list of update steps (inputs to the target
            function `f`). Entires can be `None`.
        verbose (bool, optional): Print information during the backtracking.

    Returns:
        A tuple containing
        best_cg_iter (int): The index of the entry in `steps_list` with the
            minimum function value.
        f_step (float): The minimum function value among all steps in
            `steps_list`.
    """

    if verbose:
        print("\nBacktracking cg-iterations...")

    # Target function value for all steps in `steps_list`
    f_steps_list = []
    for step in steps_list:
        if step is not None:
            f_steps_list.append(f(step))
        else:
            f_steps_list.append(float("inf"))

    # Find optimal cg-iteration
    best_cg_iter = torch.argmin(torch.Tensor(f_steps_list))

    if verbose:
        for cg_iter, f_step in enumerate(f_steps_list):
            if steps_list[cg_iter] is None:
                continue

            info = f"cg-iteration {cg_iter}, loss = {f_step:.6f}"
            info = "* " + info if cg_iter == best_cg_iter else "  " + info
            print(info)

    return best_cg_iter, f_steps_list[best_cg_iter]


def cg_efficient_backtracking(f, steps_list, verbose=False):
    """Evaluate the target function `f` for a subset of the update steps in
    `steps_list`. We start with the last item in `steps_list` and keep
    evaluating the target function as long as there is an improvement in `f`.
    Return the minimum observed (!) target function value and the corresonding
    list index.

    Args:
        f (callable): Target function: Maps a step `step` to the corresonding
            value (a float) of the target function.
        steps_list (list): Contains a list of update steps (inputs to the target
            function `f`). Entires can be `None`.
        verbose (bool, optional): Print information during the backtracking.

    Returns:
        A tuple containing
        best_cg_iter (int): The index of the entry in `steps_list` with the
            minimum observed (!) function value.
        f_step (float): The minimum function value among the considered (!)
            steps in `steps_list`.
    """

    if verbose:
        print("\nBacktracking cg-iterations...")

    # Initializations
    f_steps_list = ["not evaluated"] * len(steps_list)
    f_min = float("inf")

    # Go backwards through steps, evaluate as long as there is improvement
    for iter, step in reversed(list(enumerate(steps_list))):
        if step is None:
            continue

        # Evaluate `f` and store in list
        f_step = f(step)
        f_steps_list[iter] = f_step

        # Does `f` still decrease
        if f_step < f_min:
            f_min = f_step
            best_iter = iter
        else:  # no further improvement: `f_step >= lowest_f``
            break

    # Print information for all steps in `steps_list`
    if verbose:
        for iter, f_step in enumerate(f_steps_list):
            if steps_list[iter] is None:
                continue

            if f_step == "not evaluated":
                info = f"  cg-iteration {iter}, loss not evaluated"
            else:
                info = f"cg-iteration {iter}, loss = {f_step:.6f}"
                info = "* " + info if iter == best_iter else "  " + info
            print(info)

    return best_iter, f_steps_list[best_iter]
