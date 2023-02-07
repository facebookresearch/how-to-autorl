# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import setuptools

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "all": [
        "GPy",
        "scipy",
        "sklearn",
        "scikit-learn",
        "tabulate",
        "stable-baselines3[extra]",
        "gym",
        "torch",
        "tensorflow-gpu",
        "matplotlib",
        "sobol-seq",
        "dehb @ git+https://github.com/automl/DEHB.git@master",
        "baselines @ git+https://github.com/openai/baselines.git@master",
    ],
    "dehb": ["dehb @ git+https://github.com/automl/DEHB.git@master"],
    "pb2": ["GPy", "scipy", "sklearn"],
    "bgt": ["gpytorch", "scipy", "sklearn"],
}

setuptools.setup(
    name="autorl-sweepers",
    author="TheEimer",
    author_email="eimer@tnt.uni-hannover.de",
    description="AutoRL Sweepers for Hydra",
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    # url=url,
    # project_urls=project_urls,
    version="0",
    packages=setuptools.find_namespace_packages(include=["hydra_plugins.*"], exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "hydra-core",
        "rich",
        "hydra_colorlog",
        "hydra-submitit-launcher",
        "pandas",
        "configspace",
        "numpy",
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
