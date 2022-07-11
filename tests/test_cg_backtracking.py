"""So far, this script just runs `cg_backtracking` and
`cg_efficient_backtracking` on a toy example to make sure it runs without
throwing an error and to be able to investigate manually if it works correctly.
"""

from hessianfree.cg_backtracking import (
    cg_backtracking,
    cg_efficient_backtracking,
)

steps_list = [1.0, None, 2.7, 2.4, None, None, 7.3]


def f(step):
    """Dummy target function."""
    return step + 10


def test_cg_backtracking():

    print("\n===== Testing `cg_backtracking` =====")
    result = cg_backtracking(f, steps_list, verbose=True)
    print(f"best_iter = {result[0]}, f_best_iter = {result[1]:.6f}")

    print("\n===== Testing `cg_efficient_backtracking` =====")
    result = cg_efficient_backtracking(f, steps_list, verbose=True)
    print(f"best_iter = {result[0]}, f_best_iter = {result[1]:.6f}")


if __name__ == "__main__":
    test_cg_backtracking()
