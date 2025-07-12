from setuptools import find_packages, setup

setup(
    name="aiu-fms-testing-utils",
    version="0.0.1",
    description="Testing Utilities for use with foundation-model-stack models running on AIU",
    packages=find_packages(),
    install_requires=["torch == 2.7.1"],
)
