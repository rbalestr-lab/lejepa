from setuptools import setup, find_packages

setup(
    name="lejepa",
    version="0.0.1",
    license="MIT",
    url="https://github.com/rbalestr-lab/lejepa",
    author="Randall Balestriero",
    author_email="randallbalestriero@gmail.com",
    description="Lean Joint-Embedding Predictive Architecture (LeJEPA): Provable and Scalable Self-Supervised Learning",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "loguru>=0.7.0",
        "pytest>=7.0.0",
    ],
)
