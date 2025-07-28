#!/bin/bash
set -e

# Ensure we have the latest build tools
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel build

# Install build dependencies first
python -m pip install setuptools-scm
python -m pip install "setuptools>=65.0.0" "wheel>=0.37.0"

# Install requirements with specific flags
python -m pip install --no-build-isolation --no-use-pep517 -r requirements.txt
