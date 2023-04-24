import setuptools

setuptools.setup(
        name="torch_hyperbolic",
        version="0.0.1",
        author="Florin Ratajczak",
        description="Hyperbolic NNs and GNNs in torch",
        packages=["torch_hyperbolic"],
        install_requires=["torch>1.11", "torch-geometric>2", "torch-sparse"]
)

