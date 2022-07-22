# `PyTorchHessianFree` 

Here, we provide a PyTorch implementation of the Hessian-free optimizer as
described in [1] and [2] (references see below). The `pytorch-hessianfree`
[repo](https://github.com/fmeirinhos/pytorch-hessianfree/blob/master/hessianfree.py)
by GitHub-user `fmeirinhos` served as a starting point. For the matrix-vector
products with the Hessian or GGN, we use functionality from the BackPACK
[package](https://backpack.pt/) [3].


### Installation instructions

If you want to use the optimizer, you can download it from GitHub via `git clone
https://github.com/ltatzel/PyTorchHessianFree.git` and install it with `pip
install -e PyTorchHessianFree`. 


### MWE

```python:
"""A minimal working example using the `HessianFree` optimizer on a small neural
network and some dummy data.
"""

import torch
from hessianfree.optimizer import HessianFree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
DIM = 10

if __name__ == "__main__":

    # Set up model, loss-function and optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(DIM, DIM, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(DIM, DIM),
    ).to(DEVICE)
    loss_function = loss_function = torch.nn.MSELoss(reduction="mean")
    opt = HessianFree(model.parameters(), verbose=True)

    # Training
    for step_idx in range(5):

        # Sample dummy data, define the `forward`-function
        inputs = torch.rand(BATCH_SIZE, DIM).to(DEVICE)
        targets = torch.rand(BATCH_SIZE, DIM).to(DEVICE)

        def forward():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            return loss, outputs

        # Update the model's parameters
        opt.step(forward=forward)
```


### Structure of this repo

The repo contains three folders:
- `examples`: This folder contains a few **basic examples** demonstrating how to
use the optimizer for training neural networks and optimizing deterministic
functions (e.g. the Rosenbrock function). 
- `hessianfree`: This folder contains all the optimizer's components (e.g. the
  line search, the cg-method and preconditioners). The **Hessian-free
  optimizer** is implemented in the `optimizer.py` file.
- `tests`: Here, we **test** functionality implemented in `hessianfree`. 


### Implementation Details

- **Hessian & GGN:** Our implementation allows using either the Hessian matrix
or the GGN via the argument `curvature_opt` to the optimizer's constructor. We
chose the GGN as the default as it *tends to work much better than the Hessian
in practice* [2, p. 10]. For the matrix-vector products with these matrices, we
use functionality from the BackPACK package [3].
- **PCG:** Our implementation of the preconditioned conjugate gradient method
  features the termination criterion presented in [1, Section 4.4] via the
  argument `martens_conv_crit`. It also offers ways to deal with non-positive
directional curvature $p_i^\top A p_i \leq 0$ (note that this is a violation of
the assumption that $A$ is positive definite) via the
`nonpos_curv_option`-argument to the function `postprocess_pAp`. For example, it
allows using the absolute value of the directional curvature - this idea is
discussed in detail in [4].
- TODO: Mention all sub-section in [1, Section 4], mention `step` and `step_acc`


### Requirements

For the examples, you also need DeepOBS. TODO


### References

[1] "Deep learning via Hessian-free optimization" by James Martens. In
    Proceedings of the 27th International Conference on International Conference
    on Machine Learning (ICML), 2010. Paper available at
    https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf (accessed
    June 2022).

[2] "Training Deep and Recurrent Networks with Hessian-Free Optimization" by
    James Martens and Ilya Sutskever. Report available at
    https://www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf (accessed June
    2022).

[3] "BackPACK: Packing more into Backprop" by Felix Dangel, Frederik Kunstner
    and Philipp Hennig. In International Conference on Learning Representations,
    2020. Paper available at https://openreview.net/forum?id=BJlrF24twB
    (accessed June 2022). Python package available at
    https://github.com/f-dangel/backpack.

[4] "Identifying and attacking the saddle point problem in high-dimensional
    non-convex optimization" by Yann Dauphin, Razvan Pascanu, Caglar Gulcehre,
    Kyunghyun Cho, Surya Ganguli and Yoshua Bengio. In Advances in Neural
    Information Processing Systems, 2020. Paper available at
    https://arxiv.org/abs/1406.2572 (accessed June 2022).