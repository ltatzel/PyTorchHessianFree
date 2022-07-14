"""Implementation of preconditioners for cg."""

import torch
from backpack import backpack, extend
from backpack.extensions import SumGradSquared
from torch.nn.utils.convert_parameters import parameters_to_vector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diag_EF_backpack(model, loss_function, inputs, targets, reduction):
    """Compute the diagonal of the empirical Fisher matrix using BackPACK. This
    diagonal is the (scaled) sum of the squared gradients and can therefore be
    computed using BackPACK's `SumGradSquared()` extension (see
    https://docs.backpack.pt/en/master/extensions.html). Let `f_i` be the loss
    of the i-th sample, with gradient `g_i`. This function will return
    - `g_1^2 + ... + g_N^2` if `reduction == "sum"`,
    - `(1/N) * (g_1^2 + ... + g_N^2)` if `reduction == "mean"`

    Note: This function is less flexible than `diag_EF_autograd` (e.g. it dows
    not support L2-regularized losses), but faster. If the BackPACK-version is
    applicable, it should be preferred.

    Args:
        model (torch.nn.Module): The neural network.
        loss_function (torch.nn.Module): The loss function mapping the tuple
            `(outputs, targets)` to the loss value.
        inputs, targets (torch.Tensor): The inputs and targets for computing
            the network's output and respective loss.

    Returns:
        The diagonal of the empirical Fisher matrix as vector. Its length
        corresponds to the number of trainable parameters in `model`.
    """

    if reduction not in ["sum", "mean"]:
        raise ValueError(f"reduction {reduction} is not supported.")

    # Use BackPACK for computing the sum of the squared gradients
    model = extend(model)
    loss_function = extend(loss_function)
    loss = loss_function(model(inputs), targets)
    with backpack(SumGradSquared()):
        loss.backward()

    # Collect and convert to vector
    sum_grad_squared_list = []
    for param in model.parameters():
        if param.grad is not None:
            sum_grad_squared_list.append(param.sum_grad_squared)
    sum_grad_squared = parameters_to_vector(sum_grad_squared_list)

    if reduction == "mean":  # BackPACK already divided by `N**2`
        N = inputs.shape[0]
        sum_grad_squared = sum_grad_squared * N

    return sum_grad_squared


def diag_EF_autograd(model, loss_function, inputs, targets, reduction):
    """Compute the diagonal of the empirical Fisher matrix using autograd. This
    diagonal is the (scaled) sum of the squared gradients. Let `f_i` be the loss
    of the i-th sample, with gradient `g_i`. This function will return
    - `g_1^2 + ... + g_N^2` if `reduction == "sum"`,
    - `(1/N) * (g_1^2 + ... + g_N^2)` if `reduction == "mean"`

    Note: This function is more flexible than `diag_EF_backpack`, but slower. If
    the BackPACK-version is applicable, it should be preferred.

    Args:
        model (torch.nn.Module): The neural network.
        loss_function (torch.nn.Module): The loss function mapping the tuple
            `(outputs, targets)` to the loss value.
        inputs, targets (torch.Tensor): The inputs and targets for computing
            the network's output and respective loss.
        reduction (str): Either `"mean"` or `"sum"`, see description above.

    Returns:
        The diagonal of the empirical Fisher matrix as vector. Its length
        corresponds to the number of trainable parameters in `model`.
    """

    if reduction not in ["sum", "mean"]:
        raise ValueError(f"reduction {reduction} is not supported.")

    params_list = [p for p in model.parameters() if p.requires_grad]

    # Compute individual gradients, square and add to `sum_grad2`
    diag_EF = torch.zeros_like(parameters_to_vector(params_list))
    for (input_i, target_i) in zip(inputs, targets):
        loss_i = loss_function(model(input_i), target_i)
        grad_i = torch.autograd.grad(loss_i, params_list, retain_graph=False)
        diag_EF += parameters_to_vector(grad_i) ** 2

    # Fix scaling for reduction `"mean"`
    if reduction == "mean":
        N = inputs.shape[0]
        diag_EF = diag_EF / N

    return diag_EF


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
        A function that computes `M^-1 * x` (by using the vector representation,
        i.e. without building the matrix in memory explicitly).
    """

    def M_func(x):
        return torch.mul((diag_vec + damping) ** -exponent, x)

    return M_func


def diag_EF_preconditioner(
    model,
    loss_function,
    inputs,
    targets,
    reduction,
    damping,
    exponent=None,
    use_backpack=True,
):
    """This is simply a wrapper function calling one of the functions above to
    compute the diagonal of the empirical Fisher matrix and turning it into a
    preconditioner using `diag_to_preconditioner`. For an explanation on the
    function arguments, see above.
    """

    if use_backpack:
        diag_EF = diag_EF_backpack(
            model, loss_function, inputs, targets, reduction
        )
    else:  # use autograd
        diag_EF = diag_EF_autograd(
            model, loss_function, inputs, targets, reduction
        )

    if exponent is None:  # use default from `diag_to_preconditioner`
        M_func = diag_to_preconditioner(diag_EF, damping)
    else:
        M_func = diag_to_preconditioner(diag_EF, damping, exponent)
    return M_func


if __name__ == "__main__":

    torch.manual_seed(0)

    print(f"\nRunning on DEVICE = {DEVICE}")

    # Batch size
    N = 3

    # Parameters of the network
    D_in = 2
    D_hidden = 3
    D_out = 2

    # Data
    inputs = torch.rand(N, D_in).to(DEVICE)
    targets = torch.rand(N, D_out).to(DEVICE)

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_hidden),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Linear(D_hidden, D_hidden),
            torch.nn.ReLU(),
        ),
        torch.nn.Linear(D_hidden, D_out),
    ).to(DEVICE)

    # Loss-function
    reduction = "mean"
    loss_function = torch.nn.MSELoss(reduction=reduction)

    # Diagonal of the empirical Fisher matrix using BackPACK
    diag_EF_bp = diag_EF_backpack(
        model, loss_function, inputs, targets, reduction=reduction
    )
    print("\ndiag_EF_bp[:10] = ", diag_EF_bp[:10])

    # ... and using autograd
    diag_EF_ag = diag_EF_autograd(
        model, loss_function, inputs, targets, reduction=reduction
    )
    print("\ndiag_EF_ag[:10] = ", diag_EF_ag[:10])

    # Build preconditioner, apply to random vector
    damping = 0.1
    exponent = 0.75
    M_func = diag_to_preconditioner(diag_EF_bp, damping, exponent)
    rand_vec = torch.rand(diag_EF_bp.numel())
    Mvec = M_func(rand_vec)
    print("\nMvec[:10] = ", Mvec[:10])
