#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
]


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="OpenLockAgents",
    version="1.0",
    description="OpenLock agents for the OpenAI Gym environment",
    author="Mark Edmonds",
    author_email="mark@mjedmonds.com",
    url="https://github.com/mjedmonds/OpenLockAgents",
    packages=[
        package for package in find_packages() if package.startswith("openlockagents")
    ],
    install_requires=required,
    ext_modules=cythonize(extensions, gdb_debug=True)
)
