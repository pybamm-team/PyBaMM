import os
import sys
import subprocess
import tarfile
from pathlib import Path
from shutil import copy
import glob
from platform import python_version

try:
    from setuptools import setup, find_packages, Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        pybamm_dir = os.path.abspath(os.path.dirname(__file__))
        try:
            subprocess.run(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the KLU python module."
            )

        try:
            assert os.path.isfile("third-party/pybind11/tools/pybind11Tools.cmake")
        except AssertionError:
            print(
                "Error: Could not find "
                "third-party/pybind11/pybind11/tools/pybind11Tools.cmake"
            )
            print("Make sure the pybind11 repository was cloned in ./third-party/")
            print("See installation instructions for more information.")

        py_version = python_version()
        cmake_args = ["-DPYTHON_EXECUTABLE={}".format(sys.executable)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake for idaklu solver", "-" * 40)
        subprocess.run(["cmake", cmake_list_dir] + cmake_args, cwd=self.build_temp)

        print("-" * 10, "Building idaklu module", "-" * 40)
        subprocess.run(["cmake", "--build", "."], cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


setup(
    name="pybamm",
    version="1.0",
    description="Python Battery Mathematical Modelling.",
    long_description="description",
    long_description_content_type="text/markdown",
    url="https://github.com/pybamm-team/PyBaMM",
    include_package_data=True,
    packages=find_packages(include=("pybamm", "pybamm.*")),
    ext_modules=[Extension("idaklu", ["pybamm/solvers/c_solvers/idaklu.cpp"])],
    cmdclass={"build_ext": CMakeBuild},
    # List of dependencies
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.3",
        "pandas>=0.24",
        "anytree>=2.4.3",
        "autograd>=1.2",
        "scikit-fem>=0.2.0",
        "casadi>=3.5.0",
        "jupyter",  # For example notebooks
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
    entry_points={
        "console_scripts": [
            "pybamm_edit_parameter = pybamm.parameters_cli:edit_parameter",
            "pybamm_add_parameter = pybamm.parameters_cli:add_parameter",
            "pybamm_list_parameters = pybamm.parameters_cli:list_parameters",
        ],
    },
)
