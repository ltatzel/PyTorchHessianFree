"""Pytorch implementation of the Hessian-free optimizer."""

from warnings import warn

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product
from torch.nn.utils.convert_parameters import parameters_to_vector

from hessianfree.cg import cg
from hessianfree.cg_backtracking import cg_efficient_backtracking
from hessianfree.linesearch import simple_linesearch
from hessianfree.preconditioners import diag_EF_preconditioner
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
        cg_decay_x0=0.95,
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
            damping (float, optional): Tikhonov damping: If `0.0`, it won't get
                adapted
            cg_decay_x0: From [2, Section 10]
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
            warn("The damping is set to `0.0` and won't get adapted.")

        # Hypterparameters for cg
        if cg_max_iter is not None and cg_max_iter < 1:
            raise ValueError(f"Invalid cg_max_iter: {cg_max_iter}")
        self.cg_decay_x0 = cg_decay_x0
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

        # All computations are performed in the subspace of trainable parameters
        self._params_list = [p for p in self._params if p.requires_grad]
        self.device = self._params_list[0].device

    def step(
        self,
        forward,
        grad=None,
        mvp=None,
        M_func=None,
    ):
        """TODO

        forward: Used for everything after cg. Returns a loss-outputs-tuple
            (outputs can be `None`, but then `curvature_opt` cannot be `"ggn"`).
            You may want to check that the model is in eval mode.
        grad: The gradient of the loss function. It is used as right hand side
            in cg and in the line search. If not given, this is coputed based on
            `forward`.
            You may want to check that the model is in eval mode.
        mvp: Matrix vector product used in cg.
            You may want to check that the model is in eval mode.
        M_func: M is supposed to approximate the inverse of `A`, i.e. the
            inverse of the damped (!!) curvature matrix.
        """

        # Set state
        self.state.setdefault("x0", None)

        # ----------------------------------------------------------------------
        # Print some information
        # ----------------------------------------------------------------------
        if self.verbose:
            print("\nInformation on parameters...")

            num_params = sum(p.numel() for p in self._params)
            print("Total number of parameters: ", num_params)

            num_params = sum(p.numel() for p in self._params if p.requires_grad)
            print("Number of trainable parameters: ", num_params)

            print("Device = ", self.device)

        # ----------------------------------------------------------------------
        # Set up linear system
        # ----------------------------------------------------------------------

        # Forward pass
        loss, outputs = forward()
        init_loss = loss.item()
        if self.verbose:
            print(f"\nInitial loss = {init_loss:.6f}")

        # Evaluate the gradient
        if grad is None:
            grad = torch.autograd.grad(
                loss, self._params_list, create_graph=True, retain_graph=True
            )
            grad = parameters_to_vector(grad).detach()

        # Matrix-vector products with the curvature matrix
        curvature_opt = self._group["curvature_opt"]
        if mvp is None:
            if curvature_opt == "hessian":

                def mvp(x):
                    return self._Hv(loss, self._params_list, x)

            elif curvature_opt == "ggn":

                def mvp(x):
                    return self._Gv(loss, outputs, self._params_list, x)

        # ----------------------------------------------------------------------
        # Apply (preconditioned) cg
        # ----------------------------------------------------------------------
        damping = self._group["damping"]
        cg_max_iter = self._group["cg_max_iter"]

        # Apply cg
        x_iters, m_iters = cg(
            A=lambda x: mvp(x) + damping * x,  # Add damping
            b=-grad,
            x0=self.state["x0"],
            M=M_func,
            max_iter=cg_max_iter,
            martens_conv_crit=True,
            store_x_at_iters=None,  # Use automatic grid
            verbose=self.verbose,
        )
        step_vec = x_iters[-1]

        # Initialize the next cg-run with the decayed current solution
        self._set_x0(self.cg_decay_x0 * x_iters[-1])

        # ----------------------------------------------------------------------
        # Define target function from `forward`
        # ----------------------------------------------------------------------

        # Backup of original trainable parameters as vector
        params_vec = parameters_to_vector(self._params_list).detach()

        def tfunc(step):
            """Evaluate the target funtion that is to be minimized."""
            vector_to_trainparams(params_vec + step, self._params)
            return forward()[0].item()

        # ----------------------------------------------------------------------
        # Adapt damping (LM heuristic)
        # ----------------------------------------------------------------------
        assert x_iters[0] is not None and x_iters[-1] is not None
        if self.adapt_damping:
            self._adapt_damping(
                f_0=tfunc(x_iters[0]),  # = `init_loss` only if `x0 = 0` in cg
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

        if not self.use_linesearch:
            # Constant learning rate
            if self.verbose:
                print(f"\nConstant lr = {lr:.6f}")
            final_loss = None  # Has to be evaluated

        else:
            # Perform line search
            lr, final_loss = simple_linesearch(
                f=tfunc,
                f_grad_0=grad,
                step=step_vec,
                init_alpha=lr,
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

        # Print initial and final loss
        if self.verbose:
            if final_loss is None:
                final_loss = forward()[0].item()
            msg = f"Initial loss = {init_loss:.6f} --> "
            msg += f"final loss = {final_loss:.6f}"
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

        If a negative reduction ratio is detected, we raise a warning.

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
            print(f"  f_0    = {f_0:.6f}")
            print(f"  f_step = {f_step:.6f}")
            print(f"  m_0    = {m_0:.6f}")
            print(f"  m_step = {m_step:.6f}")
            print(f"  Reduction ratio rho = {rho:.6f}")

        # Levenberg-Marquardt heuristic for adjusting the damping constant
        if rho < 0.25:
            self._group["damping"] *= 3 / 2
        elif rho > 0.75:
            self._group["damping"] *= 2 / 3

        if self.verbose:  # Print new damping
            damping = self._group["damping"]
            print(f"  Damping is set to {damping:.6f}")

        if rho < 0:  # Bad cg-initialization
            msg = "The reduction ratio `rho` is negative. This might result in "
            msg += "a bad cg-initialization in the next step."
            warn(msg)

    def _set_x0(self, new_x0):
        """Set the "x0" value in the state dictionary to `new_x0`. This will be
        used as initialization for the cg-method.

        Args:
            new_x0 (torch.Tensor): The new value for `x0`, which is used to
                initialize the cg-method.
        """
        self.state["x0"] = new_x0

    def get_preconditioner(
        self,
        model,
        loss_function,
        inputs,
        targets,
        reduction,
        exponent=None,
        use_backpack=True,
    ):
        """This is simply a wrapper function calling `diag_EF_preconditioner`
        from `preconditioners.py`. It automatically sets the correct damping
        value currently used by the optimizer.
        """

        diag_EF_preconditioner(
            model,
            loss_function,
            inputs,
            targets,
            reduction,
            damping=self._group["damping"],
            exponent=exponent,
            use_backpack=use_backpack,
        )

    def _forward_lists(self, model, loss_func, datalist):
        """Evaluate the network's outputs, losses and mini-batch sizes for all
        mini-batches in `datalist`.
        """

        print("_forward_lists")

        losses_list = []
        outputs_list = []
        N_list = []

        for inputs, targets in datalist:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            N_list.append(targets.shape[0])
            outputs_list.append(model(inputs))
            losses_list.append(loss_func(outputs_list[-1], targets))

        return losses_list, outputs_list, N_list

    def _acc_loss_and_outputs(self, model, loss_func, datalist):
        """Accumulate the loss values and the model's outputs over all
        mini-batches in `datalist`.
        """

        print("_acc_loss_and_outputs")
        losses_list, outputs_list, N_list = self._forward_lists(
            model, loss_func, datalist
        )
        print("N_list = ", N_list)

        num_data = sum(N_list)
        loss = sum(N * l for (N, l) in zip(N_list, losses_list)) / num_data

        return loss, torch.cat(outputs_list, dim=0)

    def _acc_grad(self, model, loss_func, datalist):
        """Accumulate the gradient over all mini-batches in `datalist`."""

        print("_acc_grad")
        losses_list, _, N_list = self._forward_lists(model, loss_func, datalist)
        print("N_list = ", N_list)

        grad = torch.zeros_like(parameters_to_vector(self._params_list))
        for loss, N in zip(losses_list, N_list):
            mb_grad = torch.autograd.grad(loss, self._params_list)
            grad += N * parameters_to_vector(mb_grad).detach()

        num_data = sum(N_list)
        return grad / num_data

    def _acc_mvp(self, model, loss_func, datalist, x):
        """Accumulate the matrix-vector products with a vector `x` over a given
        list of `(inputs, targets)` pairs.
        """

        print("_acc_mvp")
        losses_list, outputs_list, N_list = self._forward_lists(
            model, loss_func, datalist
        )
        print("N_list = ", N_list)

        curvature_opt = self._group["curvature_opt"]
        mvp = torch.zeros_like(parameters_to_vector(self._params_list))

        for loss, outputs, N in zip(losses_list, outputs_list, N_list):
            if curvature_opt == "hessian":
                mvp += N * self._Hv(loss, self._params_list, x)
            elif curvature_opt == "ggn":
                mvp += N * self._Gv(loss, outputs, self._params_list, x)

        num_data = sum(N_list)
        return mvp / num_data

    def acc_step(
        self,
        model,
        loss_func,
        forward_datalist,
        grad_datalist=None,
        mvp_datalist=None,
        M_func=None,
    ):
        """TODO: Perform an optimization step but the loss/gradient/curvature
        are evaluated over a list of mini-batches.
        """

        # Set model to eval-mode?

        # Forward
        def forward():
            return self._acc_loss_and_outputs(
                model, loss_func, forward_datalist
            )

        # Try it out
        print("\n===== Calling forward() =====")
        forward()

        # Data for gradient computation
        if grad_datalist is None:
            grad_datalist = forward_datalist

        # Gradient
        print("\n===== Compute gradient =====")
        grad = self._acc_grad(model, loss_func, grad_datalist)

        # Matrix vector product
        def mvp(x):
            return self._acc_mvp(model, loss_func, mvp_datalist, x)

        # Try it out
        print("\n===== Calling mvp(x) =====")
        x = torch.rand(parameters_to_vector(self._params_list).shape)
        y = mvp(x)

        self.step(forward=forward, grad=grad, mvp=mvp, M_func=M_func)
