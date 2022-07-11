from warnings import warn

import torch


def vector_to_trainparams(vec, parameters):
    """Similar to `vector_to_parameters` from `torch.nn.utils`: Replace the
    parameters with the entries of `vec`. But here, the vector `vec` only
    contains the parameter values for the trainable parameters, i.e. those
    parameters with `requires_grad == True`. Finally, this function raises a
    warning in case not all entries of `vec` have been used.
    """

    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"`vec` should be a torch.Tensor, not {type(vec)}.")

    # Use slices of `vec` as parameter values but only if they are trainable
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        if param.requires_grad:
            param.data = vec[pointer : pointer + num_param].view_as(param).data
            pointer += num_param

    # Make sure all entries of the vector have been used (i.e. that `vec` and
    # `parameters` have the same number of elements)
    if pointer != len(vec):
        warn("Not all entries of `vec` have been used.")


def vector_to_parameter_list(vec, parameters):
    """Convert the vector `vec` to a parameter-list format matching
    `parameters`. This function is the inverse of `parameters_to_vector` from
    `torch.nn.utils`. In contrast to `vector_to_parameters`, which replaces the
    value of the parameters, this function leaves the parameters unchanged and
    returns a list of parameter views of the vector. This function raises a
    warning if not all entries of `vec` are converted.
    """

    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"`vec` should be a torch.Tensor, not {type(vec)}.")

    # Put slices of `vec` into `params_list`
    params_list = []
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        params_list.append(
            vec[pointer : pointer + num_param].view_as(param).data
        )
        pointer += num_param

    # Make sure all entries of the vector have been used (i.e. that `vec` and
    # `parameters` have the same number of elements)
    if pointer != len(vec):
        warn("Not all entries of `vec` have been used.")

    return params_list
