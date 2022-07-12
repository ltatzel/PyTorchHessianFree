"""Pytorch implementation of the Hessian-free optimizer."""

from warnings import warn

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product
from torch.nn.utils.convert_parameters import parameters_to_vector

from hessianfree.cg import cg
from hessianfree.cg_backtracking import cg_efficient_backtracking
from hessianfree.linesearch import simple_linesearch
from hessianfree.utils import vector_to_parameter_list, vector_to_trainparams


class HessianFree(torch.optim.Optimizer):
    """TODO"""

    def __init__(
        self,
        params,
        curvature_opt="ggn",
        damping=1.0,
        adapt_damping=True,
        cg_max_iter=250,
        use_cg_backtracking=True,
        lr=1.0,
        use_linesearch=True,
        verbose=False,
    ):
        """TODO

        Args:
            cg_max_iter (int, optional): The maximum number of cg-iterations.
                The default value `250` is taken from the report [1, p. 36]. If
                `None` is used, this parameter is set to the dimension of the
                linear system.
            lr (float, optional): If `use_linesearch == False`, use the constant
                learning rate, otherwise use it as initial scaling for the line
                search.
            damping (float, optional): If `0.0`, it won't get adapted
        """

        # Curvature option
        if curvature_opt not in ["hessian", "ggn"]:
            raise ValueError(f"Invalid curvature_opt = {curvature_opt}")

        # Damping
        if damping < 0.0:
            raise ValueError(f"Invalid damping = {damping}")
        self.adapt_damping = adapt_damping

        if damping == 0.0 and adapt_damping:
            self.adapt_damping = False
            warn("The damping is set to 0.0 and won't get adapted.")

        # Hypterparameters for cg
        if cg_max_iter is not None and cg_max_iter < 1:
            raise ValueError(f"Invalid cg_max_iter: {cg_max_iter}")
        self.use_cg_backtracking = use_cg_backtracking

        # Learing rate
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate lr = {lr}")
        self.use_linesearch = use_linesearch

        # Call parent class constructor
        defaults = dict(
            curvature_opt=curvature_opt,
            damping=damping,
            cg_max_iter=cg_max_iter,
            lr=lr,
        )
        super().__init__(params, defaults)

        # For now, only one parameter group is supported
        if len(self.param_groups) != 1:
            error_msg = "`HessianFree` does not support per-parameter options."
            raise ValueError(error_msg)

        self.verbose = verbose
        self._group = self.param_groups[0]
        self._params = self._group["params"]

    def step(self, eval_loss_and_outputs):
        """TODO"""

        # ----------------------------------------------------------------------
        # Print some information
        # ----------------------------------------------------------------------
        if self.verbose:
            print("\nInformation on parameters...")

            # Total number of parameters
            num_params = sum(p.numel() for p in self._params)
            print("Total number of parameters: ", num_params)

            # Number of trainable parameters
            num_params = sum(p.numel() for p in self._params if p.requires_grad)
            print("Number of trainable parameters: ", num_params)

        # ----------------------------------------------------------------------
        # Compute loss and its gradient
        # ----------------------------------------------------------------------

        # All computations are performed in the subspace of trainable parameters
        params_list = [p for p in self._params if p.requires_grad]

        # Forward pass
        loss, outputs = eval_loss_and_outputs()
        if self.verbose:
            print(f"\nInitial loss = {float(loss):.6f}")

        # Compute gradient, convert to vector, detach
        loss_grad = torch.autograd.grad(
            loss, params_list, create_graph=True, retain_graph=True
        )
        loss_grad = parameters_to_vector(loss_grad).detach()

        # ----------------------------------------------------------------------
        # Set up linear system
        # ----------------------------------------------------------------------

        curvature_opt = self._group["curvature_opt"]
        if curvature_opt == "hessian":

            def A_func(x):
                return self._Hv(loss, params_list, x)

        elif curvature_opt == "ggn":

            def A_func(x):
                return self._Gv(loss, outputs, params_list, x)

        # ----------------------------------------------------------------------
        # Apply (preconditioned) cg
        # ----------------------------------------------------------------------
        damping = self._group["damping"]
        cg_max_iter = self._group["cg_max_iter"]

        # Apply cg
        x_iters, m_iters = cg(
            A=lambda x: A_func(x) + damping * x,  # add damping
            b=-loss_grad,
            max_iter=cg_max_iter,
            martens_conv_crit=True,
            store_x_at_iters=None,  # use automatic grid
            verbose=self.verbose,
        )
        step_vec = x_iters[-1]

        # ----------------------------------------------------------------------
        # Define target function
        # ----------------------------------------------------------------------

        # Backup of original trainable parameters as vector
        params_vec = parameters_to_vector(params_list).detach()

        def tfunc(step):
            """Evaluate the target funtion that is to be minimized."""
            vector_to_trainparams(params_vec + step, self._params)
            return eval_loss_and_outputs()[0].item()

        # ----------------------------------------------------------------------
        # Adapt damping (LM heuristic)
        # ----------------------------------------------------------------------
        if self.adapt_damping:
            self._adapt_damping(
                f_0=loss.item(),
                f_step=tfunc(x_iters[-1]),
                m_0=m_iters[0],
                m_step=m_iters[-1],
            )

        # ----------------------------------------------------------------------
        # Backtracking cg-iterations
        # ----------------------------------------------------------------------
        if self.use_cg_backtracking:
            best_cg_iter, _ = cg_efficient_backtracking(
                f=tfunc,
                steps_list=x_iters,
                verbose=self.verbose,
            )
            step_vec = x_iters[best_cg_iter]

        # ----------------------------------------------------------------------
        # Line-search
        # ----------------------------------------------------------------------
        lr = self._group["lr"]

        if not self.use_linesearch:  # Constant learning rate
            lr = float(lr)
            if self.verbose:
                print(f"\nConstant lr = {lr:.6f}")
        else:  # Perform line search
            lr, _ = simple_linesearch(
                f=tfunc,
                f_grad_0=loss_grad,
                step=step_vec,
                init_alpha=float(lr),
                verbose=self.verbose,
            )

        # ----------------------------------------------------------------------
        # Parameter update
        # ----------------------------------------------------------------------

        # Update parameters
        if self.verbose:
            print(f"\nParameter update with lr = {lr:.6f}")
        new_params_vec = params_vec + lr * step_vec
        vector_to_trainparams(new_params_vec, self._params)

        # Print final loss
        if self.verbose:
            final_loss = eval_loss_and_outputs()[0]
            msg = f"Initial loss = {float(loss):.6f} --> "
            msg += f"final loss = {float(final_loss):.6f}"
            print(msg)

    @staticmethod
    def _Hv(loss, params_list, vec):
        """The Hessian-vector product from `BackPACK` [3]."""
        vec_list = vector_to_parameter_list(vec, params_list)
        Hv = hessian_vector_product(loss, params_list, vec_list)
        return parameters_to_vector(Hv).detach()

    @staticmethod
    def _Gv(loss, outputs, params_list, vec):
        """The GGN-vector product from `BackPACK` [3]."""
        vec_list = vector_to_parameter_list(vec, params_list)
        Gv = ggn_vector_product_from_plist(loss, outputs, params_list, vec_list)
        return parameters_to_vector(Gv).detach()

    def _adapt_damping(self, f_0, f_step, m_0, m_step):
        """Adapt the damping constant according to a Levenberg-Marquardt style
        heuristic [1, section 4.1]. This heuristic is based on the "agreement"
        between the actual reduction in the target function (when applying the
        update step) and the improvement predicted by the quadratic model. Note
        that this method changes the `self._group["damping"]` attribute.

        Args:
            f_0, f_step: The target function value at `0` (no update step, i.e.
                at the initial parameters) and at `step` (i.e. when applying the
                full update step).
            m_0, m_step: The value of the quadratic model used by cg at `0` (no
                update step) and at `step`.
        """

        # Compute reduction ratio `rho`
        rho = (f_step - f_0) / (m_step - m_0)
        if self.verbose:
            print("\nLM-heurisitc: Adapt damping...")
            print(f"  Reduction ratio rho = {rho:.6f}")

        # Levenberg-Marquardt heuristic for adjusting the damping constant
        if rho < 0.25:
            self._group["damping"] *= 3 / 2
        elif rho > 0.75:
            self._group["damping"] *= 2 / 3

        if self.verbose:  # Print new damping
            damping = self._group["damping"]
            print(f"  Damping is set to {damping:.6f}")

        if rho < 0:  # Bad initialization
            msg = "The reduction ratio rho is < 0. This might result in a bad "
            msg += "cg-initialization in the next step."
            warn(msg)


# # The empirical Fisher diagonal (Section 20.11.3)
# def empirical_fisher_diagonal(net, xs, ys, criterion):
#     grads = list()
#     for (x, y) in zip(xs, ys):
#         fi = criterion(net(x), y)
#         grads.append(
#             torch.autograd.grad(fi, net.parameters(), retain_graph=False)
#         )

#     vec = torch.cat(
#         [(torch.stack(p) ** 2).mean(0).detach().flatten() for p in zip(*grads)]
#     )
#     return vec


# # The empirical Fisher matrix (Section 20.11.3)
# def empirical_fisher_matrix(net, xs, ys, criterion):
#     grads = list()
#     for (x, y) in zip(xs, ys):
#         fi = criterion(net(x), y)
#         grad = torch.autograd.grad(fi, net.parameters(), retain_graph=False)
#         grads.append(torch.cat([g.detach().flatten() for g in grad]))

#     grads = torch.stack(grads)
#     n_batch = grads.shape[0]
#     return torch.einsum("ij,ik->jk", grads, grads) / n_batch
