#!/bin/bash

cd src/house_pricing
isort $1
black $1 --line-length=70
flake8 $1 --max-line-length=80 --ignore=E402
