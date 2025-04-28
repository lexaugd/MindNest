#!/usr/bin/env python
"""
MindNest setup script. Use this file to install the package.
This is a shim for compatibility with older tools.
Modern Python packaging uses pyproject.toml.
"""

from setuptools import setup

# This setup.py is a shim for compatibility with tools that don't support pyproject.toml yet.
# All package metadata is defined in pyproject.toml.
if __name__ == "__main__":
    setup() 