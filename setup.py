# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import setuptools

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dehb": ["dehb @ git+https://github.com/automl/DEHB.git@master"],
    "pb2": ["GPy==1.10.0", "scipy==1.11.1", "scikit-learn==1.3.0"],
    "bgt": ["gpytorch==1.11", "scipy==1.11.1", "scikit-learn==1.3.0"],
    "dev": ["pytest", "pytest-cov", "black", "flake8", "isort", "mypy"],
    "examples": ["stable-baselines3[extra]==2.0.0"]
}

setuptools.setup(
    name="autorl-sweepers",
    author="TheEimer",
    author_email="t.eimer@ai.uni-hannover.de",
    description="AutoRL Sweepers for Hydra",
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    version="0.1.0",
    packages=setuptools.find_namespace_packages(include=["hydra_plugins.*"], exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=[
        "hydra-core==1.3.2",
        "rich==13.4.2",
        "hydra_colorlog==1.2.0",
        "hydra-submitit-launcher==1.2.0",
        "pandas==2.0.3",
        "configspace==0.7.1",
        "numpy==1.23",
        "wandb==0.15.5",
        "deepcave==1.1.1",
        "pre-commit==3.3.3",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
