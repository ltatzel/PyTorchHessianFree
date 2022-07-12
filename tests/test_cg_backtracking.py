"""Test the cg-backtracking functions used by the `HessianFree` optimizer."""

from hessianfree.cg_backtracking import (
    cg_backtracking,
    cg_efficient_backtracking,
)

steps_list = [2.0, 1.0, None, 2.7, 2.4, None, None, 7.3]


def tfunc(step):
    """Dummy target function."""
    return step + 10


def test_cg_backtracking():
    """Apply `cg_backtracking` and `cg_efficient_backtracking` to the toy
    example defined above. Make sure that the behaviour (in particular the
    iteration returned by these functions) is as expected.
    """

    print("\n===== TEST `cg_backtracking` =====")
    best_iter, f_best_iter = cg_backtracking(
        f=tfunc,
        steps_list=steps_list,
        verbose=True,
    )
    print(f"best_iter = {best_iter}, f_best_iter = {f_best_iter:.6f}")

    # Check result
    error_msg = f"`cg_backtracking`: Unexpected result {best_iter}."
    assert best_iter == 1, error_msg

    print("\n===== TEST `cg_efficient_backtracking` =====")
    best_iter, f_best_iter = cg_efficient_backtracking(
        f=tfunc,
        steps_list=steps_list,
        verbose=True,
    )
    print(f"best_iter = {best_iter}, f_best_iter = {f_best_iter:.6f}")

    # Check result
    error_msg = f"`cg_efficient_backtracking`: Unexpected result {best_iter}."
    assert best_iter == 4, error_msg


if __name__ == "__main__":
    test_cg_backtracking()
