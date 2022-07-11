"""Proof of concept: Use cg to build an inverse of the matrix A. We start with a
simple cg implementation without preconditioning and assume `x0 = 0`.
"""

import math

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cg(A, b, x0=None, maxiter=None, tol=1e-5, atol=1e-8):

    maxiter = b.numel() if maxiter is None else min(maxiter, b.numel())
    x = torch.zeros_like(b) if x0 is None else x0

    # initialize parameters
    r = (b - A(x)).detach()
    p = r.clone()
    rs_old = (r**2).sum().item()

    # stopping criterion
    norm_bound = max([tol * torch.norm(b).item(), atol])

    def converged(rs, numiter):
        """Check whether CG stops (convergence or steps exceeded)."""
        norm_converged = norm_bound > math.sqrt(rs)
        info = numiter if norm_converged else 0
        iters_exceeded = numiter > maxiter
        return (norm_converged or iters_exceeded), info

    # Store for construction of inverse
    p_list = []
    pAp_list = []

    def build_inverse(p_list, pAp_list):
        dim = b.numel()
        A_inv = torch.zeros((dim, dim))
        for p, pAp in zip(p_list, pAp_list):
            A_inv += torch.outer(p, p) / pAp
        return A_inv

    # iterate
    iterations = 0
    while True:

        Ap = A(p).detach()
        pAp = (p * Ap).sum().item()

        # For inverse
        p_list.append(p.clone())
        pAp_list.append(pAp)

        alpha = rs_old / pAp
        x.add_(p, alpha=alpha)
        r.sub_(Ap, alpha=alpha)
        rs_new = (r**2).sum().item()
        iterations += 1

        stop, info = converged(rs_new, iterations)
        if stop:
            # print("pAp_list = ", pAp_list)
            return x, info, build_inverse(p_list, pAp_list)

        p.mul_(rs_new / rs_old)
        p.add_(r)
        rs_old = rs_new


if __name__ == "__main__":

    torch.manual_seed(0)

    dim = 3

    # Set up `A`
    A = torch.rand((dim, dim)) - 0.5
    A = A @ A.T + 0.01 * torch.eye(dim)
    print("\nA = \n", A)
    print("\nA_inv = \n", torch.linalg.inv(A))

    # Set up `b` and `x_exact`
    x = torch.rand(dim) - 0.5
    print("\nx_exact = ", x)
    b = A @ x

    # Move to `DEVICE`
    A, b, x = A.to(DEVICE), b.to(DEVICE), x.to(DEVICE)

    # Apply cg
    def A_func(x):
        return torch.matmul(A, x)

    x0 = torch.zeros_like(b).to(DEVICE)

    x_cg, info, A_inv_cg = cg(A_func, b, x0=x0)
    print("x_cg = ", x_cg)
    print("\nA_inv_cg = \n", A_inv_cg)
    print("\nA_inv_cg * b = ", torch.matmul(A_inv_cg, b), " == x_cg?")

    """Doesn't seem to work properly for larger dimensions yet:
    `x_exact` and `x_cg` are similar, but the inverses differ and surprisingly
    `A_inv_cg * b` is not the same as `x_cg` (BUG?). This has probably to do 
    with instabilities due to very small values for `pAp`. When adding a larger
    damping, the situation improves.
    """
