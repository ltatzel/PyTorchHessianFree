"""Pytorch-implementation of the preconditioned cg-algorithm."""

from math import ceil, log
from warnings import warn

import torch


def cg(
    A,
    b,
    x0=None,
    M=None,
    max_iter=None,
    tol=1e-5,
    atol=None,
    martens_conv_crit=False,
    store_x_at_iters=[],
    verbose=False,
):
    """This method implements the preconditioned conjugate gradient
    algorithm for minimizing the quadratic `0.5 x^T A x - b^T x`, with s.p.d.
    matrix `A`. Note that the minimizer of this quadratic is given by the
    solution of the linear system `A x = b`. This implementation is based on
    Algorithm 2 in Martens' report [2] with `M` denoting the inverse of `P`. It
    is the well-known preconditioned cg-algorithm with some modifications
    specifically designed for the context of Hessian-free optimization.

    Args:
        A (callable): Function implementing matrix-vector multiplication with
            the symmetric, positive definite matrix `A`.
        b (torch.Tensor): Right-hand side of the linear system `b`.
        x0 (torch.Tensor): An initial guess for the solution of the linear
            system.
        M (callable, optional): Function implementing matrix-vector
            multiplication with the preconditioning matrix. This is supposed to
            approximate the inverse of `A`. If `M == None` (default), no
            preconditioning is applied.
        max_iter (int, optional): Terminate cg after `max_iter` iterations. If
            not specified (`None`), the dimension of the linear system is used.
        tol, atol (float, optional): Terminate cg if
            `norm(residual) <= max(tol * norm(b), atol)`.
        martens_conv_crit (bool): If `True`, use Martens convergence criterion
            in addition to the tolerance-based citerion.
        store_x_at_iters (list or None): Store the cg-approximations for `x` in
            `x_iters` only in the iterations in `store_x_at_iters`. The final
            solution is always stored, even if `store_x_at_iters` is an empty
            list (the default). If `None` is given, an automatic grid is
            created.
        verbose (bool): If `True`, print out information.

    Returns:
        A tuple containing
        x_iters (list): List of approximative solutions to the linear system for
            the cg-iterations in `store_x_at_iters`. The other entries are
            `None`. The final solution for `x` is always stored.
        m_iters (list): If `martens_conv_crit == True`, this list contains the
            values of the quadratic `0.5 x^T A x - b^T x` for all cg-iterations.
    """

    if verbose:
        print("\nStarting cg...")

    # --------------------------------------------------------------------------
    # Termination criteria
    # --------------------------------------------------------------------------

    # Determine the tolerance-based termination condition
    res_bound = tol * torch.linalg.norm(b).item()
    res_bound = max([res_bound, atol]) if atol is not None else res_bound
    if verbose:
        print(f"Residual norm required for termination: {res_bound:.6e}")

    def _terminate_cg(r, x, m_iters, iter):
        """This function implements the termination conditions. We terminate cg
        if
        - Martens' convergence criterion is fulfilled
        - `max_iter` iterations have been performed
        - the residual norm diverged
        - the residual norm is within the tolerances.

        This function returns a tuple whose first entry is the "decision"
        (`True` if cg is terminated, else `False`) and the second entry (a
        string) represents the reason.
        """

        res_norm = torch.linalg.norm(r)

        # Martens' convergence criterion
        if martens_conv_crit:
            m_iters.append(0.5 * torch.dot(r - b, x))
            k = max(10, int(iter / 10))
            if k < iter:
                s_numerator = m_iters[iter] - m_iters[iter - k]
                s_denominator = m_iters[iter] - m_iters[0]
                if s_numerator / s_denominator < 1e-4:
                    return True, "Convergence (Martens)"

        # Terminate if `max_iters` iterations have been perfromed
        if iter >= max_iter:
            return True, "Number of iterations"

        # Terminate if residual norm diverged
        if torch.isnan(res_norm):
            return True, "Divergence"

        # Terminate if residual within tolerance
        if res_norm < res_bound:
            return True, "Convergence (tolerances)"

        # Don't terminate cg yet
        return False, ""

    # --------------------------------------------------------------------------
    # Dealing with non-positive curvature
    # --------------------------------------------------------------------------
    def _postprocess_pAp(pAp, iter, nonpos_curv_option="ignore"):
        """This function detects non-positive directional curvature `pAp` and
        allows to implement different options how to deal with this case. So
        far, the following options are implemented:
        - ignore: Still use the non-positive curvature `pAp`.
        - saddle-free: Take the absolute value of the directional curvature.
            This idea is taken from [4].
        """

        # If non-positive directional curvature is detected, raise a warning
        if pAp > 0:
            return pAp
        else:
            msg = f"Directional curvature pAp = {pAp:.3e} <= 0 detected in cg-"
            msg += f"iteration {iter}. This is a violation to the assumption "
            msg += "of positive definiteness."
            warn(msg)

        # How to deal with non-positive curvature
        if nonpos_curv_option == "ignore":
            return pAp
        elif nonpos_curv_option == "saddle-free":
            return abs(pAp)
        else:
            raise ValueError(f"Unknown option {nonpos_curv_option}.")

    # --------------------------------------------------------------------------
    # Store intermediate approximative solutions
    # --------------------------------------------------------------------------
    def _cg_storing_grid(max_iter, gamma=1.3):
        """This is an auxiliary function that creates a grid of iterations at
        which cg will store the respective approximative solution `x`. This
        approach and the parameter `gamma` are taken from [1, Section 4.6]: We
        include the iterations defined by `ceil(gamma^j) - 1` for `j=1, 2, ...`.

        `max_iter` is the maximum number of iterations used in the cg-method.
        `gamma` is a constant `> 1`. Values close to `1` result in a fine grid,
        values `>> 1` result in a coarse grid.
        """

        if gamma < 1.0:
            raise ValueError(f"Invalid gamma = {gamma}")

        j_max = ceil(log(max_iter + 1) / log(gamma))
        js = torch.arange(j_max + 1)  # including `j_max`

        # Remove duplicates and return list of sorted iterations
        return sorted(set((torch.ceil(gamma**js) - 1).int().tolist()))

    # --------------------------------------------------------------------------
    # Initializations
    # --------------------------------------------------------------------------

    # If not given, set maximum number of iterations and `x0`
    max_iter = b.numel() if max_iter is None else max_iter
    x0 = torch.zeros_like(b) if x0 is None else x0

    # Create grid if `None` is given, convert to dict for fast lookup
    if store_x_at_iters is None:
        store_x_at_iters = _cg_storing_grid(max_iter=max_iter)
    store_x_at_iters = {i: 0 for i in sorted(set(store_x_at_iters))}.keys()

    # Initializations
    x = x0
    x_iters = [x] if 0 in store_x_at_iters else [None]
    r = A(x0) - b
    m_iters = [0.5 * torch.dot(r - b, x0)] if martens_conv_crit else None
    y = M(r) if M is not None else r
    ry_old = torch.dot(r, y)
    p = -y

    # --------------------------------------------------------------------------
    # cg-iterations
    # --------------------------------------------------------------------------
    if verbose:
        print(f"Starting iterations (max_iter = {max_iter})...")

    iter = 1
    while True:
        if verbose:
            print(f"  cg-iteration {iter}")

        Ap = A(p).detach()
        pAp = _postprocess_pAp(torch.dot(p, Ap), iter)
        alpha = ry_old / pAp
        x = x + alpha * p
        store_x = iter in store_x_at_iters
        x_iters.append(x) if store_x else x_iters.append(None)
        r = r + alpha * Ap

        # Check termination criteria
        terminate, reason = _terminate_cg(r, x, m_iters, iter)
        if terminate:
            if verbose:
                print(reason)
            break

        y = M(r) if M is not None else r
        ry_new = torch.dot(r, y)
        beta = ry_new / ry_old
        ry_old = ry_new
        p = -y + beta * p

        # Increase iterations counter
        iter += 1

    if not store_x:
        x_iters[-1] = x  # Overwrite `None` with final solution
    return x_iters, m_iters
