#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This repo is not meant to be a package,
# but the package structure provides a clean solution to import functions

from setuptools import setup, find_packages

setup(
    name='master_thesis',
    version='0.1',
    packages=find_packages(where='.'),  # Automatically finds packages under scripts/
    include_package_data=True,
)
