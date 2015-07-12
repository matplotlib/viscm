from setuptools import setup, find_packages
import sys
import os.path

import numpy as np

# Must be one line or PyPI will cut it off
DESC = ("A colormap tool")

LONG_DESC = open("README.rst").read()

setup(
    name="viscm",
    version="0.3",
    description=DESC,
    long_description=LONG_DESC,
    author="Nathaniel J. Smith, Stefan van der Walt",
    author_email="njs@pobox.com, stefanv@berkeley.edu",
    url="https://github.com/bids/viscm",
    license="MIT",
    classifiers =
      [ "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        ],
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "colorspacious"],
    package_data={'viscm': ['examples/*']},
)
