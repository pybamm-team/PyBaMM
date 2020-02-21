import os
import sys
import subprocess
import tarfile
from shutil import copy
from platform import python_version

try:
    # wget module is required to download SUNDIALS or SuiteSparse and
    # is not a core requirement.
    import wget

    NO_WGET = False
except (ImportError, ModuleNotFoundError):
    NO_WGET = True
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
from distutils.cmd import Command


def download_extract_library(url):
    # Download and extract archive at url
    if NO_WGET:
        # The NO_WGET is set to true if the wget could not be
        # imported.
        error_msg = (
            "Could not find wget module. Please install wget module (pip install wget)."
        )
        raise ModuleNotFoundError(error_msg)
    archive = wget.download(url)
    tar = tarfile.open(archive)
    tar.extractall()


def yes_or_no(question):
    # Prompt the user with a yes or no question.
    # Only accept 'y' or 'n' characters as a valid answer.
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if len(reply) >= 1:
            if reply[0] == "y":
                return True
            if reply[0] == "n":
                return False
        print("\n")


def update_LD_LIBRARY_PATH(install_dir):
    # Look for current python virtual env and add export statement
    # for LD_LIBRARY_PATH in activate script.  If no virtual env found,
    # then the current user's .bashrc file is modified instead.

    export_statement = "export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH".format(
        install_dir
    )

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        script_path = os.path.join(venv_path, "bin/activate")
    else:
        script_path = os.path.join(os.environ.get("HOME"), ".bashrc")

    if os.getenv("LD_LIBRARY_PATH") and "{}/lib".format(install_dir) in os.getenv(
        "LD_LIBRARY_PATH"
    ):
        print("{}/lib was found in LD_LIBRARY_PATH.".format(install_dir))
        print("--> Not updating venv activate or .bashrc scripts")
    else:
        with open(script_path, "a+") as fh:
            # Just check that export statement is not already there.
            if export_statement not in fh.read():
                fh.write(export_statement)
                print(
                    "Adding {}/lib to LD_LIBRARY_PATH"
                    " in {}".format(install_dir, script_path)
                )


def install_sundials(
    sundials_src, sundials_inst, download, klu=False, force_download=False
):
    # Download the SUNDIALS library and compile it.
    # Arguments
    # ----------
    # sundials_src: str
    #     Absolute path to SUNDIALS source directory
    # sundials_inst: str
    #     Absolute path to SUNDIALS installation directory
    # download: bool
    #     Whether or not to download the SUNDIALS archive
    # klu: bool, optional
    #     Whether or not to build the SUNDIALS with KLU enabled

    pybamm_dir = os.path.abspath(os.path.dirname(__file__))

    try:
        subprocess.run(["cmake", "--version"])
    except OSError:
        raise RuntimeError("CMake must be installed to build the SUNDIALS library.")

    if download:
        question = "About to download sundials, proceed?"
        url = (
            "https://computing.llnl.gov/"
            + "projects/sundials/download/sundials-5.0.0.tar.gz"
        )
        if force_download or yes_or_no(question):
            print("Downloading SUNDIALS from " + url)
            download_extract_library(url)
        else:
            print("Exiting setup.")
            sys.exit()

    fixed_cmakelists = os.path.join(
        pybamm_dir, "scripts", "replace-cmake", "sundials-5.0.0", "CMakeLists.txt"
    )
    copy(fixed_cmakelists, os.path.join(sundials_src, "CMakeLists.txt"))

    cmake_args = [
        "-DLAPACK_ENABLE=ON",
        "-DSUNDIALS_INDEX_SIZE=32",
        "-DBUILD_ARKODE:BOOL=OFF",
        "-DEXAMPLES_ENABLE:BOOL=OFF",
        "-DCMAKE_INSTALL_PREFIX=" + sundials_inst,
    ]

    if klu:
        cmake_args.append("-DKLU_ENABLE=ON")

    # SUNDIALS are built within directory 'build_sundials' in the PyBaMM root
    # directory
    build_directory = os.path.abspath(os.path.join(pybamm_dir, "build_sundials"))
    if not os.path.exists(build_directory):
        print("\n-" * 10, "Creating build dir", "-" * 40)
        os.makedirs(build_directory)

    print("-" * 10, "Running CMake prepare", "-" * 40)
    subprocess.run(["cmake", sundials_src] + cmake_args, cwd=build_directory)

    print("-" * 10, "Building the sundials", "-" * 40)
    make_cmd = ["make", "install"]
    subprocess.run(make_cmd, cwd=build_directory)


def build_idaklu_solver(pybamm_dir):
    # Build the PyBaMM idaklu solver using cmake and pybind11.
    # Arguments
    # ---------
    # pybamm_dir: str
    #     Absolute path to PyBaMM root directory
    #
    # The CMakeLists.txt is located in the PyBaMM root directory.
    # For the build to be successful, the SUNDIALS must be installed
    # with the KLU solver enabled in pybamm_dir/sundials.

    try:
        subprocess.run(["cmake", "--version"])
    except OSError:
        raise RuntimeError("CMake must be installed to build the KLU python module.")

    try:
        assert os.path.isfile("third-party/pybind11/pybind11/tools/pybind11Tools.cmake")
    except AssertionError:
        print(
            "Error: Could not find "
            "third-party/pybind11/pybind11/tools/pybind11Tools.cmake"
        )
        print("Make sure the pybind11 repository was cloned in ./third-party/")
        print("See installation instructions for more information.")

    py_version = python_version()
    cmake_args = ["-DPYBIND11_PYTHON_VERSION={}".format(py_version)]

    print("-" * 10, "Running CMake for idaklu solver", "-" * 40)
    subprocess.run(["cmake"] + cmake_args, cwd=pybamm_dir)

    print("-" * 10, "Building idaklu module", "-" * 40)
    subprocess.run(["cmake", "--build", "."], cwd=pybamm_dir)


class InstallKLU(Command):
    """ A custom command to download and compile the SuiteSparse KLU library.
    """

    description = "Download/Compile the SuiteSparse KLU module."
    user_options = [
        # The format is (long option, short option, description).
        ("sundials-src=", None, "Path to sundials source directory"),
        ("sundials-inst=", None, "Path to sundials install directory"),
        ("suitesparse-src=", None, "Path to suitesparse source directory"),
        (
            "force-download",
            "f",
            "Whether or not to force download of SuiteSparse and Sundials libraries",
        ),
    ]
    # Absolute path to the PyBaMM root directory where setup.py is located.
    pybamm_dir = os.path.abspath(os.path.dirname(__file__))
    # Boolean flag indicating whether or not to download/install the SUNDIALS library.
    install_sundials = True

    def initialize_options(self):
        """Set default values for option(s)"""
        # Each user option is listed here with its default value.
        self.suitesparse_src = None
        self.sundials_src = None
        self.sundials_inst = None
        self.force_download = None

    def finalize_options(self):
        """Post-process options"""
        # Any unspecified option is set to the value of the 'install_all' command
        # This could be the default value if 'install_klu' is invoked on its own
        # or a user-specified value if invoked from 'install_all' with options.
        self.set_undefined_options(
            "install_all",
            ("suitesparse_src", "suitesparse_src"),
            ("sundials_src", "sundials_src"),
            ("sundials_inst", "sundials_inst"),
            ("force_download", "force_download"),
        )

        # If the SUNDIALS is already installed in sundials_inst with the KLU
        # solver enabled then do not download/install the SUNDIALS library.
        if os.path.isfile(
            os.path.join(self.sundials_inst, "lib", "libsundials_sunlinsolklu.so")
        ):
            print("Found SUNDIALS installation in {}.".format(self.sundials_inst))
            print("Not installing SUNDIALS.")
            self.install_sundials = False

        # If the SuiteSparse source dir was provided as a command line option
        # then check that it actually contains the Makefile.
        # Else, SuiteSparse must be downloaded.
        if self.suitesparse_src:
            self.suitesparse_src = os.path.abspath(self.suitesparse_src)
            klu_makefile = os.path.join(self.suitesparse_src, "KLU", "Makefile")
            assert os.path.exists(klu_makefile), "Could not find {}.".format(
                klu_makefile
            )
            self.download_suitesparse = False
        else:
            self.download_suitesparse = True
            self.suitesparse_src = os.path.join(self.pybamm_dir, "SuiteSparse-5.6.0")

        # If the SUNDIALS source dir was provided as a command line option
        # then check that it actually contains the CMakeLists.txt.
        # Else, the SUNDIALS must be downloaded.
        if self.sundials_src:
            self.sundials_src = os.path.abspath(self.sundials_src)
            CMakeLists = os.path.join(self.sundials_src, "CMakeLists.txt")
            assert os.path.exists(CMakeLists), "Could not find {}.".format(CMakeLists)
            self.download_sundials = False
        else:
            self.download_sundials = True
            self.sundials_src = os.path.join(self.pybamm_dir, "sundials-5.0.0")

    def run(self):
        """Functionality for the install_klu command.
        1. Download/build SuiteSparse
        2. Download/build SUNDIALS with KLU
        3. Build python KLU module with pybind11
        """
        try:
            subprocess.run(["make", "--version"])
        except OSError:
            raise RuntimeError(
                "Make must be installed to compile the SuiteSparse KLU module."
            )

        if self.download_suitesparse:
            question = "About to download SuiteSparse, proceed?"
            url = (
                "https://github.com/DrTimothyAldenDavis/"
                + "SuiteSparse/archive/v5.6.0.tar.gz"
            )
            if self.force_download or yes_or_no(question):
                print("Downloading SuiteSparse from " + url)
                download_extract_library(url)
            else:
                print("Exiting setup.")
                sys.exit()

        # The SuiteSparse KLU module has 4 dependencies:
        # - suitesparseconfig
        # - amd
        # - COLAMD
        # - btf
        print("-" * 10, "Building SuiteSparse_config", "-" * 40)
        make_cmd = ["make"]
        build_dir = os.path.join(self.suitesparse_src, "SuiteSparse_config")
        subprocess.run(make_cmd, cwd=build_dir)

        print("-" * 10, "Building SuiteSparse KLU module dependencies", "-" * 40)
        make_cmd = ["make", "library"]
        for libdir in ["AMD", "COLAMD", "BTF"]:
            build_dir = os.path.join(self.suitesparse_src, libdir)
            subprocess.run(make_cmd, cwd=build_dir)

        print("-" * 10, "Building SuiteSparse KLU module", "-" * 40)
        build_dir = os.path.join(self.suitesparse_src, "KLU")
        subprocess.run(make_cmd, cwd=build_dir)

        if self.install_sundials:
            install_sundials(
                self.sundials_src,
                self.sundials_inst,
                self.download_sundials,
                klu=True,
                force_download=self.force_download,
            )
        build_idaklu_solver(self.pybamm_dir)


class InstallODES(Command):
    """ A custom command to install scikits.ode with pip, as well as its main dependency the
    SUNDIALS library.
    """

    description = "Installs scikits.odes using pip."
    user_options = [
        # The format is (long option, short option, description).
        ("sundials-src=", None, "Path to sundials source dir"),
        ("sundials-inst=", None, "Path to sundials install directory"),
        (
            "force-download",
            "f",
            "Whether or not to force download of SuiteSparse and Sundials libraries",
        ),
    ]
    # Absolute path to the PyBaMM root directory where setup.py is located.
    pybamm_dir = os.path.abspath(os.path.dirname(__file__))
    # Boolean flag indicating whether or not to download/install the SUNDIALS library.
    install_sundials = True

    def initialize_options(self):
        """Set default values for option(s)"""
        # Each user option is listed here with its default value.
        self.sundials_src = None
        self.sundials_inst = None
        self.force_download = None

    def finalize_options(self):
        """Post-process options"""
        # Any unspecified option is set to the value of the 'install_all' command
        # This could be the default value if 'install_odes' is invoked on its own
        # or a user-specified value if invoked from 'install_all' with options.

        # If option specified the check dir exists
        self.set_undefined_options(
            "install_all",
            ("sundials_src", "sundials_src"),
            ("sundials_inst", "sundials_inst"),
            ("force_download", "force_download"),
        )

        # If the installation directory sundials_inst already exists, then it is
        # assumed that the SUNDIALS library is already installed in there and
        # the library is not installed.
        if os.path.exists(self.sundials_inst):
            print(
                "Found SUNDIALS installation directory {}.".format(self.sundials_inst)
            )
            print("Not installing SUNDIALS.")
            self.install_sundials = False

        # If the SUNDIALS source dir was provided as a command line option
        # then check that it actually contains the CMakeLists.txt.
        # Else, the SUNDIALS must be downloaded.
        if self.sundials_src:
            self.sundials_src = os.path.abspath(self.sundials_src)
            CMakeLists = os.path.join(self.sundials_src, "CMakeLists.txt")
            assert os.path.exists(CMakeLists), "Could not find {}.".format(CMakeLists)
            self.download_sundials = False
        else:
            self.download_sundials = True
            self.sundials_src = os.path.join(self.pybamm_dir, "sundials-5.0.0")

    def run(self):
        """Functionality for the install_odes command.
        1. Download/build SUNDIALS
        2. Update virtual env activate script or .bashrc
           to add the SUNDIALS to LD_LIBRARY_PATH.
        3. Install scikits.odes using pip
        """

        if self.install_sundials:
            # Download/build SUNDIALS
            install_sundials(
                self.sundials_src,
                self.sundials_inst,
                self.download_sundials,
                force_download=self.force_download,
            )

        update_LD_LIBRARY_PATH(self.sundials_inst)

        # At the time scikits.odes is pip installed, the path to the sundials
        # library must be contained in an env variable SUNDIALS_INST
        # see https://scikits-odes.readthedocs.io/en/latest/installation.html#id1
        os.environ["SUNDIALS_INST"] = self.sundials_inst
        env = os.environ.copy()
        subprocess.run(["pip", "install", "scikits.odes"], env=env)


class InstallAll(Command):
    """ Install both scikits.odes and KLU module.
        This command is the combination of the two commands
        'install_odes' (InstallODES)
    and
        'install_klu' (InstallKLU)
    """

    user_options = [
        ("sundials-src=", None, "Absolute path to sundials source dir"),
        ("sundials-inst=", None, "Absolute path to sundials install directory"),
        ("suitesparse-src=", None, "Absolute path to SuiteSparse root directory"),
        (
            "force-download",
            "f",
            "Whether or not to force download of SuiteSparse and Sundials libraries",
        ),
    ]

    # Absolute path to the PyBaMM root directory where setup.py is located.
    pybamm_dir = os.path.abspath(os.path.dirname(__file__))

    def initialize_options(self):
        """Set default values for option(s)"""
        # Each user option is listed here with its default value.
        self.sundials_src = None
        self.sundials_inst = os.path.join(self.pybamm_dir, "sundials")
        self.suitesparse_src = None
        self.force_download = None

    def finalize_options(self):
        """Post-process options"""
        if self.sundials_src:
            print("Using SUNDIALS source directory {}".format(self.sundials_src))
        if self.suitesparse_src:
            print("Using SuiteSparse source directory {}".format(self.sundials_src))

    def run(self):
        """Install scikits.odes and KLU module"""
        self.run_command("install_klu")
        self.run_command("install_odes")


# Load text for description and license
with open("README.md") as f:
    readme = f.read()


def load_version():
    # Read version number from file
    try:
        import os

        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, "pybamm", "version"), "r") as f:
            version = f.read().strip().split(",")
        return ".".join([str(int(x)) for x in version])
    except Exception as e:
        raise RuntimeError("Unable to read version number (" + str(e) + ").")


setup(
    cmdclass={
        "install_odes": InstallODES,
        "install_klu": InstallKLU,
        "install_all": InstallAll,
    },
    name="pybamm",
    version=load_version() + ".post4",
    description="Python Battery Mathematical Modelling.",
    long_description=readme,
    long_description_content_type="text/markdown",
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
