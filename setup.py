from setuptools import setup, find_packages

setup(
        name="torch_hyperbolic",
        version="0.0.1",
        author="Florin Ratajczak",
        description="Hyperbolic NNs and GNNs in torch",
        packages=["torch_hyperbolic", "torch_hyperbolic.nn", "torch_hyperbolic.nn"],
        install_requires=["torch>=1.11", "torch-geometric>=2"]
)

