try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# Load text for description and license
with open("README.md") as f:
    readme = f.read()


# Read version number from file
def load_version():
    try:
        import os

        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, "pybamm", "version"), "r") as f:
            version = f.read().strip().split(",")
        return ".".join([str(int(x)) for x in version])
    except Exception as e:
        raise RuntimeError("Unable to read version number (" + str(e) + ").")


setup(
    name="pybamm",
    version=load_version(),
    description="Python Battery Mathematical Modelling.",
    long_description=readme,
    url="https://github.com/pybamm-team/PyBaMM",
    include_package_data=True,
    packages=find_packages(include=("pybamm", "pybamm.*")),
    package_data={
        "pybamm": [
            "./version",
            "../input/parameters/lithium-ion/*.csv",
            "../input/parameters/lithium-ion/*.py",
            "../input/parameters/lead-acid/*.csv",
            "../input/parameters/lead-acid/*.py",
        ]
    },
    # List of dependencies
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.0",
        "pandas>=0.24",
        "anytree>=2.4.3",
        "autograd>=1.2",
        "scikit-fem>=0.2.0",
        "casadi>=3.5.0",
        "jupyter",  # For example notebooks
        "python-Levenshtein>=0.12.0",
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
            "black",  # For code style auto-formatting
        ],
    },
)
