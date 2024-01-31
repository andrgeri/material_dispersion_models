import os
from setuptools import find_packages, setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

install_requires = [
    "numpy",
    "scipy"
]

setup(
    name = "material_dispersion_models",
    version = "0.1",
    author = "Andrea Gerini",
    author_email = "andrea.gerini@u-paris.fr",
    description = "Material dispersion models",
    license = "MIT",
    keywords = "electromagnetism, optical_frequencies, refractive_index",
    url = "https://github.com/andrgeri/FDFD_mode_solver",
    python_requires=">=3.10",
    packages = find_packages(exclude=[]),
    long_description = read('README.md'),
    classifiers = [
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)