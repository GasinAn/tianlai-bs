from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include

setup(
    name='dnb',
    ext_modules=cythonize('dnb.pyx'),
    include_dirs=[get_include()],
    zip_safe=False,
)
