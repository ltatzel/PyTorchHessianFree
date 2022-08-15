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
    """This class implements the Hessian-free optimizer as described in [1] and
    [2] in Pytorch.
    """

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
        """The constructor creates an instance of the `HessianFree` optimizer.

        Args:
            params (iterable): An iterable of `torch.Tensor`s that represents
                the parameters that are to be optimized. So far, only one
                parameter group is supported.
            curvature_opt (str): The `HessianFree` optimizer uses a local
                quadratic model of the loss landscape (and minimizes this
                quadratic with the cg-method). This model requires curvature
                information, which, in our implementation, is either the Hessian
                matrix (`curvature_opt == "hessian"`) or the generalized Gauss-
                Newton matrix (`curvature_opt == "ggn"`). As recommended in [1,
                Section 4.2] and [2, e.g. p. 10], the default is the symmetric
                positive semidefinite GGN.
            damping (float): The optimizer uses Tikhonov damping (see [1,
                Section 4.1]), i.e. a scalar multiple of the identity matrix is
                added to the curvature matrix when the cg-method is applied.
                This damping parameter is kept constant only if `adapt_damping`
                is `False`, otherwise it may be modified.
            adapt_damping (bool): If `adapt_damping` is `True`, the proposed
                damping parameter `damping` is adapted according to a
                Levenberg-Marquardt style heuristic, see [1, Section 4.1].
            cg_max_iter (int or None): The maximum number of cg-iterations. The
                default value is taken from [2, Section 8.7]. If `None` is
                given, the dimension of the linear system is used.
            cg_decay_x0 (float): As suggested in [1, Section 4.5], the cg-method
                is initialized with the cg-"solution" from the last step. [2,
                Section 10] suggests multiplying this by a scalar constant
                `cg_decay_x0`.
            use_cg_backtracking (bool): If `True`, cg will return not only the
                final "solution" to the linear system but also intermediate
                "solutions" for a subset of the iterations. The set of potential
                update steps is later searched for an "ideal" candidate. This
                approach is described in [1, Section 4.6].
            lr (float): The learning rate. The update step is scaled by this
                scalar. If `use_linesearch` is `True`, it is used as the initial
                candidate.
            use_linesearch (bool): If `use_linesearch` is `True`, the update
                step is iteratively scaled back until a "sufficient" improvement
                (Armijo condition) in the loss value is achieved.
            verbose (bool): Print information during the computations.
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

    # step #####################################################################
    def step(
        self,
        forward,
        grad=None,
        mvp=None,
        M_func=None,
        test_deterministic=False,
    ):
        """Perform a parameter update step.

        Args:
            forward (callable): This function returns a `(loss, outputs)`-tuple,
                where `loss` is the target function value. Here is a pseudo-code
                example of the training loop:
                ```
                for step_idx in range(num_steps):

                    inputs, targets = get_minibatch_data()

                    def forward():
                        outputs = model(inputs)
                        loss = loss_function(outputs, targets)
                        return loss, outputs

                    opt.step(forward=forward)
                ```
                The `outputs` value is needed only when the GGN is used as the
                curvature matrix. If `curvature_opt == "hessian"`, the return
                value for `outputs` can be set to `None`.
            grad (torch.Tensor or None): The gradient of the loss with respect
                to the trainable parameters. If this is `None`, it will be
                computed from the `forward` function. This gradient is used as
                a right-hand side in the cg-method and in the Armijo condition
                in the line search.
            mvp (callable or None): This function represents the matrix-vector
                product with the curvature matrix, i.e. it maps a vector `x` to
                `B * x`, where `B` is a curvature matrix with respect to the
                trainable parameters. If this `mvp` is `None`, it is computed
                from the `forward` function.
            M_func (callable or None): This function represents the matrix-
                vector product with a preconditioner matrix. This is supposed to
                approximate multiplication with the inverse of `A` (in cg), i.e.
                the inverse of the damped (!) curvature matrix. The method
                `get_preconditioner` sets up such a function.
            test_deterministic (bool): If this is set to `True`, it is checked
                wheter two independent computations of the loss and the
                matrix-vector product (applied to a random vector), yield the
                same result. This is important because if the model has "random"
                components (e.g. uses `torch.nn.Dropout`), this will lead to
                "inconsistent" matrix-vector products in cg implying "changing"
                quadratic models which might lead to unpredictable behaviour. It
                is recommended to at least perform this test (which slightly
                increases the computational costs of the step) at least once
                (e.g. in the very first step).
        """

        # Use state to exchange and track information over multiple steps
        state = self.state
        state.setdefault("x0", None)

        state.setdefault("init_losses", [])
        state.setdefault("final_losses", [])
        state.setdefault("dampings", [])
        state.setdefault("cg_reasons", [])
        state.setdefault("num_cg_iters", [])
        state.setdefault("best_cg_iters", [])
        state.setdefault("learning_rates", [])

        # ----------------------------------------------------------------------
        # Print some information
        # ----------------------------------------------------------------------
        if self.verbose:
            print("\nInformation on parameters...")

            num_params = sum(p.numel() for p in self._params)
            print("  Total number of parameters: ", num_params)

            num_params = sum(p.numel() for p in self._params if p.requires_grad)
            print("  Number of trainable parameters: ", num_params)

            print("  Device = ", self.device)

        # ----------------------------------------------------------------------
        # Set up linear system
        # ----------------------------------------------------------------------

        # Test if the behavior of `forward` is determinsitic
        if test_deterministic:
            self._test_forward_determinisitc(forward)

        # Forward pass
        loss, outputs = forward()
        init_loss = loss.item()
        if self.verbose:
            print(f"\nInitial loss = {init_loss:.6f}")
        state["init_losses"].append(init_loss)

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

        # Test if the behavior of `mvp` is determinsitic
        if test_deterministic:
            self._test_mvp_deterministic(mvp)

        # ----------------------------------------------------------------------
        # Apply (preconditioned) cg
        # ----------------------------------------------------------------------
        damping = self._group["damping"]
        state["dampings"].append(damping)
        cg_max_iter = self._group["cg_max_iter"]

        # Only store the initial and final solution (i.e. use `[0]`); if cg-
        # backtracking is used, create an automated grid of iterations (`None`).
        store_x_at_iters = None if self.use_cg_backtracking else [0]

        # Apply cg
        x_iters, m_iters, cg_reason = cg(
            A=lambda x: mvp(x) + damping * x,  # Add damping
            b=-grad,
            x0=state["x0"],
            M=M_func,
            max_iter=cg_max_iter,
            martens_conv_crit=True,
            store_x_at_iters=store_x_at_iters,
            verbose=self.verbose,
        )
        state["cg_reasons"].append(cg_reason)
        state["num_cg_iters"].append(len(x_iters) - 1)  # `x0` also in list
        step_vec = x_iters[-1]

        # Initialize the next cg-run with the decayed current solution (not the
        # "backtracked" one, see [1, Section 4.6]).
        self._set_x0(self.cg_decay_x0 * x_iters[-1])

        # ----------------------------------------------------------------------
        # Define target function from `forward`
        # ----------------------------------------------------------------------

        # Backup of original trainable parameters as vector
        params_vec = parameters_to_vector(self._params_list).detach()

        @torch.no_grad()
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
            state["best_cg_iters"].append(best_cg_iter)
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
        state["learning_rates"].append(lr)

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
            state["final_losses"].append(final_loss)

            msg = f"Initial loss = {init_loss:.6f} --> "
            msg += f"final loss = {final_loss:.6f}"
            print(msg)

        # Return the final mini-batch loss
        return final_loss

    def _test_forward_determinisitc(self, forward):
        """Test if the behavior of the `forward` function is deterministic: Call
        this function two times and check if the retuned values are identical.
        If this is not the case, a warning is raised.

        Args:
            forward (callable): A function that performs the forward-pass. It
                returns a tuple `(loss, outputs)` of `torch.Tensor`s. `outputs`
                can be `None`.

        Raises:
            Warning if non-deterministic behavior is detected.
        """

        if self.verbose:
            print("\nTest deterministic behavior of `forward`...")
        deterministic = True

        # Compute losses and outputs
        loss_1, outputs_1 = forward()
        loss_2, outputs_2 = forward()

        # Check outputs
        if outputs_1 is not None and outputs_2 is not None:
            if not torch.allclose(outputs_1, outputs_2):
                if self.verbose:
                    print("  Test outputs: failed")
                deterministic = False
            else:
                if self.verbose:
                    print("  Test outputs: passed")

        # Check loss values
        if not torch.allclose(loss_1, loss_2):
            if self.verbose:
                print("  Test loss values: failed")
            deterministic = False
        else:
            if self.verbose:
                print("  Test loss values: passed")

        if not deterministic:
            msg = "Non-determinisitc behaviour detected. Consider setting your "
            msg += "model to evaluation mode, i.e. `model.eval()`."
            warn(msg)
        else:
            if self.verbose:
                print("  All tests passed")

    def _test_mvp_deterministic(self, mvp):
        """Test if the behavior of the `mvp` function is deterministic: Call
        this function two times and check if the retuned values are identical.
        If this is not the case, a warning is raised.

        Args:
            mvp (callable): `mvp(x)` computes the matrix-vector product with the
                vector `x`.

        Raises:
            Warning if non-deterministic behavior is detected.
        """

        if self.verbose:
            print("\nTest deterministic behavior of `mvp`...")

        # Compute matrix-vector product
        x = torch.randn_like(parameters_to_vector(self._params_list))
        x = x.to(self.device)
        mvp_1 = mvp(x)
        mvp_2 = mvp(x)

        # Check matrix-vector products
        if not torch.allclose(mvp_1, mvp_2):
            if self.verbose:
                print("  Test mvps: failed")

            # Raise warning
            msg = "Non-determinisitc behaviour detected. Consider setting your "
            msg += "model to evaluation mode, i.e. `model.eval()`."
            warn(msg)
        else:
            if self.verbose:
                print("  Test mvps: passed")
                print("  All tests passed")

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
        heuristic [1, section 4.1], [2, Section 8.5]. This heuristic is based on
        the "agreement" between the actual reduction in the target function
        (when applying the update step) and the improvement predicted by the
        quadratic model. Note that this method changes the
        `self._group["damping"]` attribute.

        Args:
            f_0, f_step: The target function value at `0` (no update step, i.e.
                at the initial parameters) and at `step` (i.e. when applying the
                full update step).
            m_0, m_step: The value of the quadratic model used by cg at `0` (no
                update step) and at `step`.

        Raises:
            Warning if a negative reduction ratio is detected.
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

    # acc_step #################################################################
    def acc_step(
        self,
        model,
        loss_func,
        loss_datalist,
        grad_datalist=None,
        mvp_datalist=None,
        M_func=None,
        reduction="mean",
        test_deterministic=False,
    ):
        """Perform an optimization step, where the loss-values (used e.g. in
        the line search), gradient and curvature are each evaluated over a list
        of mini-batches. These lists may differ! In this regard, this method is
        more flexible than `step` and its "iterative" computations (the results
        are accumulated over the chunks of data) allow to use very large batch
        sizes. However, due to more sequential work, this method is slower than
        `step`. If `step` is applicable, it should therefore be preferred. Also
        note that, so far, all quantities (loss, gradient, mvp) are computed
        independently without exchanging information. This results in redundant
        work, e.g. if all quantities are computed on the same data, the same
        forward pass is executed multiple times.

        This method is balically a wrapper for `step` that creates the
        `forward`- function, the gradient and the `mvp`-function automatically
        based on the `model`, `loss_func` and data lists.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            loss_datalist (list): A list of `(inputs, targets)`-tuples used
                by for the computation of the loss. `inputs` and `targets` are
                `torch.Tensor`s.
            grad_datalist (list or None): A list of `(inputs, targets)`-tuples
                used for the computation of the gradient. If this is `None` (the
                default), the `loss_datalist` is used.
            mvp_datalist (list or None): A list of `(inputs, targets)`-tuples
                used for the computation of the matrix-vector products. If this
                is `None` (the default), the `loss_datalist` is used.
            M_func (callable or None): The preconditioner for cg. This is
                supposed to be an approximation of the inverse of the damped (!)
                curvature matrix, see the `step`-method for details.
            reduction (str): The reduction method used by the loss function. Let
                the individual per-sample loss contributions be denoted by l_i.
                If the loss_function is a sum over these contributions, use
                `"sum"`; if it is an average, i.e. (1/N) * (l_1 + ... + l_N),
                use `"mean"`. To make sure, it is recommended to test the
                reduction with the `test_reduction`-method.
            test_deterministic (bool): Test whether the loss and the `mvp`
                function yield deterministic results, see the `step`-method for
                details.
        """

        # If not given, set the data lists to `loss_datalist`
        if grad_datalist is None:
            grad_datalist = loss_datalist

        if mvp_datalist is None:
            mvp_datalist = loss_datalist

        curvature_opt = self._group["curvature_opt"]

        # Forward
        def forward():
            return (
                self._acc_loss(model, loss_func, loss_datalist, reduction),
                None,  # outputs are set to `None`
            )

        # Gradient
        grad = self._acc_grad(model, loss_func, grad_datalist, reduction)

        # Matrix-vector product
        def mvp(x):
            return self._acc_mvp(
                model, loss_func, mvp_datalist, curvature_opt, reduction, x
            )

        # Compute the optimization step
        return self.step(
            forward=forward,
            grad=grad,
            mvp=mvp,
            M_func=M_func,
            test_deterministic=test_deterministic,
        )

    @staticmethod
    def _acc(
        model,
        loss_func,
        datalist,
        device,
        with_grad,
        init_result,
        eval_mb,
        reduction,
    ):
        """This function allows to accumulate some quantity `result` (the loss,
        gradient or mvp in our case) over multiple mini-batches.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            datalist (list): A list of `(inputs, targets)`-tuples used
                for the evaluation of the mini-batch results. `inputs` and
                `targets` are `torch.Tensor`s.
            device (torch.device): `inputs` and `targets` are moved to this
                device before the forward pass is applied.
            with_grad (bool): If `True`, build the graph for the backpropagation
                when performing the forward pass. The resulting loss and outputs
                are given to `eval_mb`.
            init_result: `results` will be initialized with this value. It has
                to be compatible with the output of `eval_mb`.
            eval_mb (callable): This function accepts two inputs: A mini-
                batch loss value and the mini-batch outputs.
            reduction (str): Either `"mean"` or `"sum"`. The result is updated
                using the `eval_mb`-function as follows:
                - `results += eval_mb(...)` if `reduction == "sum"`
                - `results += (N / num_data) * eval_mb(...)` if `reduction ==
                  "mean"`, where `N` is the mini-batch size and `num_data` is
                  the total number of datapoints over all mini-batches.

        Returns:
            The accumulated result `result`.
        """

        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid reduction {reduction}")

        # Initialize result and total number of data points
        result = init_result
        num_data = 0

        # Accumulate results using the `eval_mb` function
        for inputs, targets in datalist:
            N = targets.shape[0]
            num_data += N
            inputs, targets = inputs.to(device), targets.to(device)

            def forward_pass():
                """Perform the forward pass"""
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                return loss, outputs

            # Forward pass with or without building the graph
            if with_grad:
                loss, outputs = forward_pass()
            else:
                with torch.no_grad():
                    loss, outputs = forward_pass()

            # Compute result on mini-batch and add to result
            mb_result = eval_mb(loss, outputs)
            if reduction == "mean":
                result += N * mb_result
            else:
                result += mb_result

        # Return result
        return result / num_data if reduction == "mean" else result

    def _acc_loss(self, model, loss_func, datalist, reduction):
        """Accumulate the loss.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            datalist (list): A list of `(inputs, targets)`-tuples used
                for the evaluation of the loss. `inputs` and `targets` are
                `torch.Tensor`s.
            reduction (str): Either `"mean"` or `"sum"`. The returned loss is
                the sum of
                - all mini-batch loss-values if `reduction == "sum"`. This
                  results in the sum of the individual per-data loss-values.
                - all mini-batch loss-values scaled by `N / num_data`, where `N`
                  is the mini-batch size and `num_data` is the total number of
                  datapoints over all mini-batches. This results in the average
                  of the individual per-data loss-values.

        Returns:
            The accumulated loss-value.
        """

        def eval_mb_loss(loss, outputs):
            """Return the mini-batch loss."""
            return loss.detach()

        return self._acc(
            model,
            loss_func,
            datalist,
            device=self.device,
            with_grad=False,
            init_result=0.0,
            eval_mb=eval_mb_loss,
            reduction=reduction,
        )

    def _acc_grad(self, model, loss_func, datalist, reduction):
        """Accumulate the gradient.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            datalist (list): A list of `(inputs, targets)`-tuples used
                for the evaluation of the gradient. `inputs` and `targets` are
                `torch.Tensor`s.
            reduction (str): Either `"mean"` or `"sum"`. The returned gradient
                is the sum of
                - all mini-batch gradients if `reduction == "sum"`. This
                  results in the sum of the individual per-data gradients.
                - all mini-batch gradients scaled by `N / num_data`, where `N`
                  is the mini-batch size and `num_data` is the total number of
                  datapoints over all mini-batches. This results in the average
                  of the individual per-data gradients.

        Returns:
            The accumulated gradient vector.
        """

        init_grad = torch.zeros_like(parameters_to_vector(self._params_list))

        def eval_mb_grad(loss, outputs):
            """Compute the mini-batch gradient for `loss`."""
            mb_grad = torch.autograd.grad(loss, self._params_list)
            return parameters_to_vector(mb_grad).detach()

        return self._acc(
            model,
            loss_func,
            datalist,
            device=self.device,
            with_grad=True,
            init_result=init_grad,
            eval_mb=eval_mb_grad,
            reduction=reduction,
        )

    def _acc_mvp(self, model, loss_func, datalist, curvature_opt, reduction, x):
        """Accumulate the matrix-vector product.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            datalist (list): A list of `(inputs, targets)`-tuples used
                for the evaluation of the mvp. `inputs` and `targets` are
                `torch.Tensor`s.
            curvature_opt (str): Either `"ggn"` or `"hessian"`.
            reduction (str): Either `"mean"` or `"sum"`. The returned matrix-
                vector product is the sum of
                - all mini-batch matrix-vector products if `reduction == "sum"`.
                  This results in the sum of the individual per-data matrix-
                  vector products.
                - all mini-batch matrix-vector products scaled by
                  `N / num_data`, where `N` is the mini-batch size and
                  `num_data` is the total number of datapoints over all mini-
                  batches. This results in the average of the individual per-
                  data matrix-vector products.
            x (torch.Tensor): The matrix-vector product is applied to this
                vector.

        Returns:
            The accumulated gradient vector.
        """

        init_mvp = torch.zeros_like(parameters_to_vector(self._params_list))

        def eval_mb_mvp(loss, outputs):
            """Compute the matrix-vector product with `x`."""
            if curvature_opt == "hessian":
                return self._Hv(loss, self._params_list, x).detach()
            elif curvature_opt == "ggn":
                return self._Gv(loss, outputs, self._params_list, x).detach()

        return self._acc(
            model,
            loss_func,
            datalist,
            device=self.device,
            with_grad=True,
            init_result=init_mvp,
            eval_mb=eval_mb_mvp,
            reduction=reduction,
        )

    # misc #####################################################################
    def test_reduction(self, model, loss_func, datalist, reduction):
        """This is a test method to make sure that the loss-function and the
        specified reduction match. More precisely, we use `datalist` and call
        the accumulation functions. We then convert the data list to a single
        tuple `(ref_inputs, ref_targets)`. Both representations have to lead to
        the same results for the loss, gradient and matrix-vector product.

        Args:
            model (torch.nn.Module): The neural network mapping the `inputs`
                contained in `datalist` to `outputs`.
            loss_func (torch.nn.Module): The loss function mapping the tuple
                `(outputs, targets)` to the loss value.
            datalist (list): A list of `(inputs, targets)`-tuples used to
                compute the loss value, gradient and matrix-vector product. This
                list can be small: Two mini-batches are enough for testing
                purposes.
            reduction (str): The reduction method used by the loss function. Let
                the individual per-sample loss contributions be denoted by
                `l_i`. If the loss_function is a sum over these contributions,
                use `"sum"`; if it is an average, i.e. `(1/N) * (l_1 + ... +
                l_N)`, use `"mean"`.

        Raises:
            Exeption if the loss-function and the reduction do not match.
        """

        if self.verbose:
            print(f"\nTest reduction {reduction}...")

        # Check the data list
        error_msg = "This test is only meaningful for a data list with at "
        error_msg += "least two entries."
        assert len(datalist) > 1, error_msg

        # Sample random vector for testing the matrix-vector product
        x = torch.randn_like(parameters_to_vector(self._params_list))
        x = x.to(self.device)

        # ----------------------------------------------------------------------
        # Compute loss, gradient, matrix-vector product on `datalist`
        # ----------------------------------------------------------------------

        curvature_opt = self._group["curvature_opt"]
        acc_loss = self._acc_loss(model, loss_func, datalist, reduction)
        acc_grad = self._acc_grad(model, loss_func, datalist, reduction)
        acc_mvp = self._acc_mvp(
            model, loss_func, datalist, curvature_opt, reduction, x
        )

        # ----------------------------------------------------------------------
        # Compute loss, gradient, matrix-vector product on reference data
        # ----------------------------------------------------------------------

        # Turn datalist into two tensors: `ref_inputs` and `ref_targets`
        inputs_list = []
        targets_list = []
        for inputs, targets in datalist:
            inputs_list.append(inputs)
            targets_list.append(targets)
        ref_inputs = torch.cat(inputs_list, dim=0).to(self.device)
        ref_targets = torch.cat(targets_list, dim=0).to(self.device)

        # Forward pass
        ref_outputs = model(ref_inputs)
        ref_loss = loss_func(ref_outputs, ref_targets)

        ref_grad = parameters_to_vector(
            torch.autograd.grad(ref_loss, self._params_list, create_graph=True)
        ).detach()

        if curvature_opt == "ggn":
            ref_mvp = self._Gv(ref_loss, ref_outputs, self._params_list, x)
        elif curvature_opt == "hessian":
            ref_mvp = self._Hv(ref_loss, self._params_list, x)

        # ----------------------------------------------------------------------
        # Tests: Compare both approaches
        # ----------------------------------------------------------------------

        # Test tolerances
        RTOL = 1e-3
        ATOL = 1e-6

        check_quantities = [
            ("loss values", ref_loss, acc_loss),
            ("gradients", ref_grad, acc_grad),
            ("mvps", ref_mvp, acc_mvp),
        ]

        tests_passed = True

        for quantity, ref, acc in check_quantities:

            # Perform check `ref` == `acc`
            if not torch.allclose(acc, ref, rtol=RTOL, atol=ATOL):
                if self.verbose:
                    print(f"  Test {quantity}: failed")
                tests_passed = False
            else:  # Results match
                if self.verbose:
                    print(f"  Test {quantity}: passed")

        if not tests_passed:
            error_msg = f"Inconsistent results for reduction {reduction}. "
            error_msg += "This could also be the result of non-deterministic "
            error_msg += "behavior."
            raise RuntimeError(error_msg)
        else:
            if self.verbose:
                print("  All tests passed")

    def get_preconditioner(
        self,
        model,
        loss_func,
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
            loss_func,
            inputs,
            targets,
            reduction,
            damping=self._group["damping"],
            exponent=exponent,
            use_backpack=use_backpack,
        )
