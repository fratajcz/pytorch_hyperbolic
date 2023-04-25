from setuptools import setup, find_packages

setup(
        name="torch_hyperbolic",
        version="0.0.3",
        author="Florin Ratajczak",
        description="Hyperbolic NNs and GNNs in torch",
        packages=find_packages(exclude=["tests", "docs"]),
        install_requires=["torch>=1.11", "torch-geometric>=2"]
)
