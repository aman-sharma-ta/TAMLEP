import os
import sys
from setuptools import setup, find_packages,PEP420PackageFinder

ROOT=sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
SRC=os.path.join(ROOT,'src')

with open('README.md','r') as f:
    data=f.read()

setup(
    name='house_price_model',
    version='0.3',
    description='house pricing module',
    author='Aman',
    author_email='aman.sharma@tigeranalytics.com',
    long_description=data,
    package_dir={"": "src"},
    packages=PEP420PackageFinder.find(where=str(SRC))
)

