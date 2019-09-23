try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# Load text for description and license
with open("README.md") as f:
    readme = f.read()

setup(
    name="pybamm",
    description="Python Battery Mathematical Modelling.",
    long_description=readme,
    url="https://github.com/pybamm-team/PyBaMM",
    # include_package_data=True,
    packages=find_packages(include=('pybamm', 'pybamm.*')),
    package_data={'pybamm': [
        '../input/parameters/lithium-ion/*.csv',
        '../input/parameters/lithium-ion/*.py',
        '../input/parameters/lead-acid/*.csv',
        '../input/parameters/lead-acid/*.py',
    ]},
    # List of dependencies
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.0",
        "pandas>=0.23",
        "anytree>=2.4.3",
        "autograd>=1.2",
        "scikit-fem>=0.2.0"
        # Note: Matplotlib is loaded for debug plots, but to ensure pybamm runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        # Should not be imported
        "matplotlib>=2.0",
    ],
    extras_require={
        "docs": ["sphinx>=1.5", "guzzle-sphinx-theme"],  # For doc generation
        "dev": [
            "flake8>=3",  # For code style checking
            "jupyter",  # For documentation and testing
        ],
    },
)
