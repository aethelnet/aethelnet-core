from setuptools import setup, find_packages

setup(
    name="aethelnet-core",
    version="1.0.0",
    description="Aethelnet Core: Continuous-Time LGNN & Headless Python SDK",
    author="Auratic Systems Prime",
    license="AGPL-3.0",
    url="https://aethelburg.network",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.3",
        "networkx>=3.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        # aethelnet/__init__.py imports engine.py, which defines a FastAPI app
        "fastapi>=0.90.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "httpx>=0.23.0"],
    },
    python_requires=">=3.10",
)
