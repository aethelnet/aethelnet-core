from setuptools import setup, find_packages

setup(
    name="aethelnet-core",
    version="1.0.0",
    description="Continuous Liquid Graph Neural Network (LGNN) Core Engine",
    author="Aethelnet",
    license="AGPL-3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.3",
        "networkx>=3.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
    python_requires=">=3.9",
)
