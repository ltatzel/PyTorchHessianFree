# `PyTorchHessianFree` 

Here, we provide a PyTorch implementation of the Hessian-free optimizer as
described in [1] and [2]. The `pytorch-hessianfree`
[repo](https://github.com/fmeirinhos/pytorch-hessianfree/blob/master/hessianfree.py)
by GitHub-user `fmeirinhos` served as a starting point. For the matrix-vector
products with the Hessian or GGN, we use functionality from the BackPACK package
[3].



## Implementation Details

### General
- Our implementation allows using either the Hessian matrix or the GGN via the
argument `curvature_opt` to the optimizer's constructor. We chose the GGN as the
default as it *tends to work much better than the Hessian in practice* [2, p.
10]. For the matrix-vector products with these matrices, we use functionality
from the BackPACK package [3].

### PCG
- Our implementation of the preconditioned conjugate gradient method is based on
Algorithm 2 in [2, page 9].
- It features the termination criterion presented in Chapter 4 [2, Equation (2)]
via the argument `martens_conv_crit`. 
- It extends Algorithm 2 by detecting and offering ways to deal with
non-positive directional curvature $p_i^\top A p_i \leq 0$ (note that this is a
violation of the assumption that $A$ is positive definite) via the
`nonpos_curv_option`-argument to the function `postprocess_pAp`. For example, it
allows using the absolute value of the directional curvature - this idea is
discussed in detail in [3].

### Damping
- TODO



## References

[1] "Deep learning via Hessian-free optimization" by James Martens. In
    Proceedings of the 27th International Conference on International Conference
    on Machine Learning (ICML), 2010. Paper available at
    https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf (accessed
    June 2022).

[2] "Training Deep and Recurrent Networks with Hessian-Free Optimization" by
    James Martens and Ilya Sutskever. Report available at
    https://www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf (accessed
    June 2022).

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