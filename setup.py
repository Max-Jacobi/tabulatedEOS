from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy


setup(
    name="tabulatedEOS",
    version="1.0",
    author="Maximilian Jacobi",
    author_email="mjacobi@theorie.ikp.physik.tu-darmstadt.de",
    packages=["tabulatedEOS"],
    license='MIT',
    include_package_data=True,
    install_requires=[
        "numpy",
        "h5py",
        "alpyne @ git+https://github.com/fguercilena/alpyne@master",
    ],
    python_requires='>=3.7',
    description="Routines to interpolate inside tabulated EOS tables given in the ET format",
)
