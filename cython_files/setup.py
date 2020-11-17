from setuptools import setup
from Cython.Build import cythonize

import numpy as np


import os

os.environ["CC"] = "g++-10"    # Use g++-10 as compiler
#os.environ["CXX"] = "g++-10"

setup(
    ext_modules = cythonize("compute_perturbed_A.pyx", annotate=True),
    include_dirs=[np.get_include(),]
)