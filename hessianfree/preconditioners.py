"""Implementation of preconditioners for cg."""

import torch
from backpack import backpack, extend
from backpack.extensions import SumGradSquared, DiagGGNExact
from torch.nn.utils.convert_parameters import parameters_to_vector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sum_grad_squared_backpack(model, loss_function, inputs, targets):
    """Compute the sum of the squared gradients (this is the diagonal of the
    empirical Fisher) using BackPACK. Note, that the result depends on the
    "scaling" of the overall function (see the `SumGradSquared()` extension,
    https://docs.backpack.pt/en/master/extensions.html): Let f_i be the loss
    of the i-th sample, with gradient g_i. This function will return the sum of
    the squared
    - [g_1, ..., g_N] if the loss is a sum, f_1 + ... + f_N,
    - [g_1/N, ..., g_n/N] if the loss is a mean, (1/N) * (f_1 + ... + f_N)

    Note: This function is less flexible than `sum_grad_squared_autograd` (e.g.
    it dows not support L2-regularized losses), but faster. If it is applicable,
    it should be preferred over the Autograd version below.

    Args:
        model (torch.nn.Module): The neural network.
        loss_function (torch.nn.Module): The loss function mapping the tuple
            `(outputs, targets)` to the loss value.
        inputs, targets (torch.Tensor): The inputs and targets for computing
            the loss.

    Returns:
        The sum of the squared gradients as vector. Its length corresponds to
        the number of trainable parameters in `model`.
    """

    # Use BackPACK for computing the sum of the squared gradients
    model = extend(model)
    loss_function = extend(loss_function)
    loss = loss_function(model(inputs), targets)
    with backpack(SumGradSquared()):
        loss.backward()

    # Collect and convert to vector
    squared_grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            squared_grad_list.append(param.sum_grad_squared)
    return parameters_to_vector(squared_grad_list)


def sum_grad_squared_autograd(model, loss_function, inputs, targets, reduction):
    """Compute the sum of the squared gradients (this is the diagonal of the
    empirical Fisher) using Autograd. Note, that the result depends on the
    "scaling" of the overall function: Let f_i be the loss of the i-th sample,
    with gradient g_i. This function will return the sum of the squared
    - [g_1, ..., g_N] if `reduction == "sum"`,
    - [g_1/N, ..., g_N/N] if `reduction == "mean"`

    Note: This function is more flexible than `sum_grad_squared_backpack`, but
    slower. If the BackPACK-version is applicable, it should be preferred.

    Args:
        model (torch.nn.Module): The neural network.
        loss_function (torch.nn.Module): The loss function mapping the tuple
            `(outputs, targets)` to the loss value.
        inputs, targets (torch.Tensor): The inputs and targets for computing
            the loss.
        reduction (str): Either `"mean"` or `"sum"`, see description above.

    Returns:
        The sum of the squared gradients as vector. Its length corresponds to
        the number of trainable parameters in `model`.
    """

    if reduction not in ["sum", "mean"]:
        raise ValueError(f"reduction {reduction} is not supported.")

    N = inputs.shape[0]
    params_list = [p for p in model.parameters() if p.requires_grad]

    # Compute individual gradients, square and add to `sum_grad2`
    sum_grad2 = torch.zeros_like(parameters_to_vector(params_list))
    for (input_i, target_i) in zip(inputs, targets):
        loss_i = loss_function(model(input_i), target_i)
        grad_i = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        sum_grad2 += parameters_to_vector(grad_i) ** 2

    # Fix scaling for reduction `"mean"`
    if reduction == "mean":
        sum_grad2 = sum_grad2 / N**2

    return sum_grad2


def diag_F_autograd(model, loss_function, inputs, targets, reduction):
    if reduction not in ["sum", "mean"]:
        raise ValueError(f"reduction {reduction} is not supported.")

    N = inputs.shape[0]
    params_list = [p for p in model.parameters() if p.requires_grad]

    # Compute individual gradients, square and add to `sum_grad2`
    diag_F = torch.zeros_like(parameters_to_vector(params_list))
    for (input_i, target_i) in zip(inputs, targets):
        loss_i = loss_function(model(input_i), target_i)
        grad_i = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        diag_F += parameters_to_vector(grad_i) ** 2

    # Fix scaling for reduction `"mean"`
    if reduction == "mean":
        diag_F = diag_F / N

    return diag_F


def diag_to_preconditioner(diag_vec, damping, exponent=0.75):
    """Turn a diagonal matrix represented by `diag_vec` into a preconditioner,
    as described in [1, Section 4.7]. The preconditioning matrix is given by
    `M = (diag_matrix + damping * I) ** exponent`, where `I` is the identity
    matrix.

    Args:
        diag_vec (torch.Tensor): A vector that represents a diagonal matrix.
        damping (float): Scalar damping (denoted by lambda in [1, Section 4.7])
        exponent (float): Scalar exponent (denoted by alpha in [1, Section 4.7])

    Returns:
        A function that computes M^-1 * x (by using the vector representation,
        i.e. without building the matrix in memory explicitly).
    """

    def M_func(x):
        return torch.mul((diag_vec + damping) ** -exponent, x)

    return M_func


def compute_GGN(model, loss_function, inputs, targets, reduction):

    N = inputs.shape[0]

    params_list = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params_list)

    outputs = model(inputs)
    num_outputs = outputs.shape[1]

    G = torch.zeros((num_params, num_params))
    for n in range(N):
        target = targets[n, :]
        output = outputs[n, :]

        # Compute the Jacobian
        Jac = torch.zeros((num_outputs, num_params))
        for i in range(num_outputs):
            out = output[i]
            Jac[i, :] = parameters_to_vector(
                torch.autograd.grad(out, params_list, retain_graph=True)
            )

        # Compute the Hessian
        loss = loss_function(target, output)
        loss_grad = torch.autograd.grad(
            loss, output, create_graph=True, retain_graph=True
        )[0]
        Hess = torch.zeros((num_outputs, num_outputs))
        for i in range(num_outputs):
            Hess[i, :] = torch.autograd.grad(
                loss_grad[i], output, retain_graph=True
            )[0]

        # Update GGN
        G = G + Jac.T @ Hess @ Jac
    return G / N if reduction == "mean" else G


def ggn_diag_backpack(model, loss_function, inputs, targets):

    # Use BackPACK for computing the diagonal of the GGN
    model = extend(model)
    loss_function = extend(loss_function)
    loss = loss_function(model(inputs), targets)
    with backpack(DiagGGNExact()):
        loss.backward()

    # Collect and convert to vector
    diag_ggn_list = []
    for param in model.parameters():
        if param.grad is not None:
            diag_ggn_list.append(param.diag_ggn_exact)

    return parameters_to_vector(diag_ggn_list)


def empirical_fisher(model, loss_function, inputs, targets, reduction):

    N = inputs.shape[0]
    params_list = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params_list)

    # Compute individual gradients, square and add to `sum_grad2`
    F = torch.zeros((num_params, num_params))
    for (input_i, target_i) in zip(inputs, targets):
        loss_i = loss_function(model(input_i), target_i)
        grad_i = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        grad_i = parameters_to_vector(grad_i)
        F = F + torch.outer(grad_i, grad_i)

    # Fix scaling for reduction `"mean"`
    if reduction == "mean":
        F = F / N

    return F

    # outputs = model(inputs)
    # outputs_flat = outputs.flatten()
    # # print("outputs_flat = ", outputs_flat)
    # num_outputs = outputs.numel()

    # # Compute the Jacobian
    # Jac = torch.zeros((num_outputs, num_params))
    # for i, output in enumerate(outputs_flat):
    #     Jac[i, :] = parameters_to_vector(
    #         torch.autograd.grad(output, params_list, retain_graph=True)
    #     )
    # # print("Jac = \n", Jac)

    # # Compute the loss Hessian (with respect to `outputs`)
    # print("outputs.shape = ", outputs.shape)
    # print("targets.shape = ", targets.shape)
    # loss = loss_function(outputs, targets)
    # print("loss = ", loss)
    # loss_grad = torch.autograd.grad(
    #     loss,
    #     outputs,
    #     create_graph=True,
    #     retain_graph=True,
    # )[0].flatten()
    # print("loss_grad = ", loss_grad)

    # Hess = torch.zeros((num_outputs, num_outputs))
    # for i, g in enumerate(loss_grad):
    #     Hess[i, :] = parameters_to_vector(
    #         torch.autograd.grad(g, outputs, retain_graph=True)
    #     )
    # print("Hess = ", Hess)

    # print("loss_grad = ", loss_grad)
    # print("params_list = ", params_list)
    # Jac = torch.autograd.grad(outputs, params_list)
    # print("Jac = ", Jac)


END_IDX = 5

if __name__ == "__main__":

    torch.manual_seed(0)

    print(f"\nRunning on DEVICE = {DEVICE}")

    # Problem parameters
    N = 3  # Batch size
    D_in = 2
    D_hidden = 3
    D_out = 2

    # Data
    inputs = torch.rand(N, D_in).to(DEVICE)
    targets = torch.rand(N, D_out).to(DEVICE)

    # Model
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(D_in, D_hidden),
    #     torch.nn.ReLU(),
    #     torch.nn.Sequential(
    #         torch.nn.Linear(D_hidden, D_hidden),
    #         torch.nn.ReLU(),
    #     ),
    #     torch.nn.Linear(D_hidden, D_out),
    # ).to(DEVICE)
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_out),
    ).to(DEVICE)

    reduc = "mean"
    loss_function = torch.nn.MSELoss(reduction=reduc)

    ggn = compute_GGN(model, loss_function, inputs, targets, reduction=reduc)
    ggn_diag_ag = torch.diag(ggn)
    print("\ndiag(GGN) my implementation: \n", ggn_diag_ag)

    ggn_diag_bp = ggn_diag_backpack(model, loss_function, inputs, targets)
    print("diag(GGN) backpack: \n", ggn_diag_bp)

    print("Close? ", torch.allclose(ggn_diag_ag, ggn_diag_bp))

    # Empirical Fisher
    F = empirical_fisher(model, loss_function, inputs, targets, reduction=reduc)
    print("\ndiag(F) = ", torch.diag(F))

    diag_F_ag = diag_F_autograd(
        model, loss_function, inputs, targets, reduction=reduc
    )
    print("diag_F_ag = ", diag_F_ag)
    # print("ratio = ", torch.div(ggn_diag_ag, sum_grad_squared))

    #################################################################

    # # Freeze parameters of first layer --> some parameters not trainable
    # first_layer = next(model.children())
    # for param in first_layer.parameters():
    #     param.requires_grad = False

    # Loss function
    # def get_l2_loss(l2_factor):
    #     l2_loss = 0.0
    #     for p in model.parameters():
    #         l2_loss += p.pow(2).sum()
    #     return 0.5 * l2_factor * l2_loss

    # ADD_L2_LOSS = False
    # REDUCTION = "mean"
    # if ADD_L2_LOSS:

    #     def loss_function(inputs, targets):
    #         loss = torch.nn.MSELoss(reduction=REDUCTION)(inputs, targets)
    #         loss += get_l2_loss(0.1)
    #         return loss

    # else:
    #     loss_function = torch.nn.MSELoss(reduction=REDUCTION)

    # REDUCTION = "sum"
    # print("\nREDUCTION = ", REDUCTION)

    # Loss function
    # loss_function = torch.nn.MSELoss(reduction=REDUCTION)

    # # With BackPACK
    # bp_grad2_vec = sum_grad_squared_backpack(
    #     model, loss_function, inputs, targets
    # )
    # # print("\nbp_grad2_vec.shape = ", bp_grad2_vec.shape)
    # print("bp_grad2_vec[:END_IDX] = ", bp_grad2_vec[:END_IDX])

    # # With Autograd
    # ag_grad2_vec = sum_grad_squared_autograd(
    #     model, loss_function, inputs, targets, reduction=REDUCTION
    # )
    # # print("\nag_grad2_vec.shape = ", ag_grad2_vec.shape)
    # print("ag_grad2_vec[:END_IDX] = ", ag_grad2_vec[:END_IDX])

    # # Same result
    # print("\nclose ? ", torch.allclose(bp_grad2_vec, ag_grad2_vec))

    # # Use BackPACK for computing the diag of the GGN
    # model.zero_grad()
    # model = extend(model)
    # loss_function = extend(loss_function)
    # loss = loss_function(model(inputs), targets)
    # with backpack(DiagGGNExact()):
    #     loss.backward()

    # # Collect and convert to vector
    # diag_ggn_list = []
    # for param in model.parameters():
    #     if param.grad is not None:
    #         diag_ggn_list.append(param.diag_ggn_exact)
    # diag_GGN = parameters_to_vector(diag_ggn_list)
    # print("\ndiag_GGN[:END_IDX] = ", diag_GGN[:END_IDX])

    # ratio_vec = torch.div(diag_GGN, ag_grad2_vec)
    # print("\nratio_vec[:END_IDX] = ", ratio_vec[:END_IDX])

    # # Preconditioner
    # M_func = diag_to_preconditioner(bp_grad2_vec, damping=0.1)
    # print("\nM_func = ", M_func)

    # x_vec = torch.ones_like(ag_grad2_vec)
    # M_func_x = M_func(x_vec)
    # print("M_func_x.shape = ", M_func_x.shape)
    # print("M_func_x[:5] = ", M_func_x[:5])
