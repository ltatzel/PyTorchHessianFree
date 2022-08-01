# `PyTorchHessianFree` 

Here, we provide a PyTorch implementation of the Hessian-free optimizer as
described in [1] and [2] (see below). This project is currently still being
developed, so changes may be made at any time.

**Core idea:** At each step, the  optimizer computes a local quadratic
approximation of the target function and uses the conjugate gradient (cg) method
to approximate its minimum (the Newton step). This method only requires access
to matrix-vector products with the curvature matrix, which can be done without
creating this matrix in memory explicitly. This makes the Hessian-free optimizer
applicable for large problems with high-dimensional parameter spaces (e.g.
training neural networks).

**Credits:** The `pytorch-hessianfree`
[repo](https://github.com/fmeirinhos/pytorch-hessianfree/blob/master/hessianfree.py)
by GitHub-user `fmeirinhos` served as a starting point. For the matrix-vector
products with the Hessian or GGN, we use functionality from the BackPACK
[package](https://backpack.pt/) [3].

**Table of contents:**
1. [Installation instructions](#installation)
2. [Example](#example)
3. [Structure of this repo](#structure)
4. [Implementation details](#details)
5. [Contributing](#contributing)
6. [References](#references)

---

## 1. Installation instructions <a name="installation"></a>

If you want to use the optimizer, you can download the repo from GitHub via `git
clone https://github.com/ltatzel/PyTorchHessianFree.git`. Then, navigate to the
project folder `cd PyTorchHessianFree` and install it with `pip install -e .`.

Additional requirements for the **tests and examples** can be installed via `pip
install -e ".[tests]"` and `pip install -e ".[examples]"` respectively. For
running the tests, execute `pytest` from the repo's root directory.


## 2. Example <a name="example"></a>

```python
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
    loss_function = torch.nn.MSELoss()
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


## 3. Structure of this repo <a name="structure"></a>

The repo contains three folders:
- `hessianfree`: This folder contains all the optimizer's components (e.g. the
  line search, the cg-method and preconditioners). The **Hessian-free
  optimizer** itself is implemented in the `optimizer.py` file.
- `tests`: Here, we **test** functionality implemented in `hessianfree`. 
- `examples`: This folder contains a few **basic examples** demonstrating how to
use the optimizer for training neural networks (using the `step` and `acc_step`
method) and optimizing deterministic functions (e.g. the Rosenbrock function). 


## 4. Implementation details <a name="details"></a>

- **Hessian & GGN:** Our implementation allows using either the Hessian matrix
  or the GGN as curvature matrix via the argument `curvature_opt` to the
  optimizer's constructor. As recommended in [1, Section 4.2] and [2, e.g. p.
  10], the default is the symmetric positive semidefinite GGN. For the
  matrix-vector products with these matrices, we use functionality from the
  BackPACK package [3].

- **Damping:** As described in [1, Section 4.1], Tikhonov-damping can be used to
  avoid overly large steps. Our implementation also features the
  Levenberg-Marquardt style heuristic for adjusting the damping parameter - it
  can be turned on and off via the `adapt_damping` switch.

- **PCG:** Our implementation of the preconditioned conjugate gradient method
  features the termination criterion presented in [1, Section 4.4] via the
  argument `martens_conv_crit`. 
  
  It also offers ways to deal with non-positive directional curvature 
  $p_i^\top A p_i \leq 0$ (note that this is a violation of the assumption that
  $A$ is positive definite) via the `nonpos_curv_option`-argument to the
  `postprocess_pAp` function. For example, it allows using the absolute value of
  the directional curvature - this idea is discussed in detail in [4]. 
  
  As suggested in [1, Section 4.5], we use the cg- "solution" from the last step
  as a starting point for the next one. Via the argument `cg_decay_x0` to the
  optimizer's constructor, this initial search direction can be scaled by a
  constant. The default is `0.95` as in [2, Section 10].

  The `get_preconditioner`-method implements the preconditioner suggested in [1,
  Section 4.7]: The diagonal of the empirical Fisher matrix. 

- **CG-backtracking & line search:** When cg-backtracking is used, the
  `cg`-method will return not only the final "solution" to the linear system but
  also intermediate "solutions" for a subset of the iterations. This grid of
  iterations is generated using the approach from [1, Section 4.6]. In a
  subsequent step, the set of potential update steps is searched for an "ideal"
  candidate. 
  
  Next, this update step is iteratively scaled back by the line search until the
  target function is decreased "significantly" (Armijo condition). This approach
  is described in [2, Section 8.8]. 
  
  Both these modules are optional and can be turned on and off via the switches
  `use_cg_backtracking` and `use_linesearch`.

- **Computing parameter updates:** Our Hessian-free optimizer offers two methods
  for computing parameter updates: `step` and `acc_step`. 

  The former one, which is also used in the example above, only has one required
  argument: the `forward`-function. This represents the target function and all
  relevant quantities needed by the optimizer (e.g. the gradient and curvature
  information) are deduced from this function. 
  
  You may want to use the latter method `acc_step` if you run out of memory when
  training your neural network model using `step` or if you want to evaluate the
  target function value (the loss), gradient and curvature on different data
  sets. The `acc_step` method allows you to specify (potentially different)
  lists of data for these three quantities. It evaluates e.g. the gradient only
  on one list entry (i.e. one mini-batch) at a time and `acc`umulates the
  individual gradients automatically. This iterative approach slows down the
  computations but enables us to work with very large data sets. A basic example
  can be found
  [here](https://github.com/ltatzel/PyTorchHessianFree/blob/740bd80346873a75f904bbba15f0737403a3d511/examples/run_small_nn_acc.py).


## 5. Contributing <a name="contributing"></a>

I would be very grateful for any feedback! If you have questions, a feature
request, found a bug or have comments on how to improve the code, please don't
hesitate to reach out to me.


## 6. References <a name="references"></a>

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
  