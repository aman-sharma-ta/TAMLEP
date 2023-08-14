import os

from setuptools import find_packages, setup

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

with open(os.path.join(ROOT, "README.md")) as f:
    ld = f.read()


setup(
    name="house_price_module",
    version="0.3",
    package_dir={"": "src"},
    description="House pricing module",
    long_description=ld,
    author="Aman",
    author_email="aman.sharma@tigeranalytics.com",
    packages=find_packages(where=str(SRC)),
    requires=["pandas", "numpy", "matplotlib", "scikit-learn"]
)
