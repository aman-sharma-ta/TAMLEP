import os
from setuptools import setup,PEP420PackageFinder

ROOT=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(ROOT,'src')

with open('README.md','r') as f:
    data=f.read()

setup(
    name='house_pricing',
    version='0.3',
    description='house pricing module',
    author='Aman',
    author_email='aman.sharma@tigeranalytics.com',
    long_description=data,
    package_dir={"": "src"},
    packages=PEP420PackageFinder.find(where=str(SRC))
)

