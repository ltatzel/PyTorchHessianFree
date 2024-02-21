from setuptools import find_packages, setup

DEEPOBS_LINK = (
    "git+https://github.com/fsschneider/DeepOBS.git@develop#egg=deepobs"
)

setup(
    name="hessianfree",
    version="0.1",
    description="PyTorch implementation of the Hessian-free optimizer",
    author="Lukas Tatzel",
    url="https://github.com/ltatzel/PyTorchHessianFree",
    packages=find_packages(),
    install_requires=[
        "backpack-for-pytorch>=1.5.0,<2.0.0",
        "torch>=1.11.0",
    ],
    extras_require={
        "tests": [  # install with `pip install -e ".[tests]"`
            "pytest>=7.1.2",
        ],
        "examples": [  # install with `pip install -e ".[examples]"`
            "deepobs @ " + DEEPOBS_LINK,
        ],
    },
)
