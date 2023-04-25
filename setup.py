from setuptools import setup, find_packages

setup(
        name="torch_hyperbolic",
        version="0.0.2",
        author="Florin Ratajczak",
        description="Hyperbolic NNs and GNNs in torch",
        packages=find_packages(exclude=["tests", "docs"]),
        install_requires=["torch==2.0.0+cpu", "torch-geometric>=2"]
)

