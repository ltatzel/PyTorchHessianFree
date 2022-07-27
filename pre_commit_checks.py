"""This is an auxiliary script that can be executed before a git commit. It runs
the tests using `pytest`, the formatters `black` and `isort` and makes sure that
all examples run withour throwing an error.
"""

import os
import subprocess

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE_DIR, "examples")

RUN_TESTS = True
RUN_FORMATTERS = True
RUN_EXAMPLES = True

if __name__ == "__main__":

    # Tests
    if RUN_TESTS:
        print("\nRunning tests...")
        subprocess.run("pytest")

    # Formatters
    if RUN_FORMATTERS:
        print("\nRunning formatters...")
        subprocess.run("black .")
        subprocess.run("isort .")

    # Examples
    if RUN_EXAMPLES:
        print("\nRunning examples...")
        for fn in os.listdir(EXAMPLES_DIR):
            if fn.startswith("run_") and fn.endswith(".py"):

                print(f"  Running {fn}")
                completed_process = subprocess.run(
                    "python " + os.path.join(EXAMPLES_DIR, fn),
                    stdout=subprocess.DEVNULL,  # suppress output
                    stderr=subprocess.DEVNULL,  # suppress warnings & errors
                )
                completed_process.check_returncode()
