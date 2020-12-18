#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as req:
    reqs = req.readlines()

setup(
    name="topotool",
    version="0.0.1",
    description="Topology tool for FreeIPA deployments using ansible-freeipa with power of graph theory",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Tibor Dudlák",
    author_email="tibor.dudlak@gmail.com",
    url="https://github.com/Tiboris/topotool",
    license="Apache License 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=['wheel'],
    install_requires=reqs,
    scripts=["scripts/topotool"],
)
