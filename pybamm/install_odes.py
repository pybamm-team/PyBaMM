import os
import tarfile
from os.path import join, isfile
import argparse
import sys
import logging
import subprocess

from pybamm.util import root_dir as pybamm_dir

try:
    # wget module is required to download SUNDIALS or SuiteSparse.
    import wget

    NO_WGET = False
except ModuleNotFoundError:
    NO_WGET = True


def download_extract_library(url, directory):
    # Download and extract archive at url
    if NO_WGET:
        error_msg = (
            "Could not find wget module."
            " Please install wget module (pip install wget)."
        )
        raise ModuleNotFoundError(error_msg)
    archive = wget.download(url, out=directory)
    tar = tarfile.open(archive)
    tar.extractall(directory)


def install_sundials():
    # Download the SUNDIALS library and compile it.
    logger = logging.getLogger("scikits.odes setup")
    sundials_version = "5.1.0"

    try:
        subprocess.run(["cmake", "--version"])
    except OSError:
        raise RuntimeError("CMake must be installed to build SUNDIALS.")

    directory = join(pybamm_dir(), "scikits.odes")
    os.makedirs(directory, exist_ok=True)
    url = (
        "https://computing.llnl.gov/"
        + "projects/sundials/download/sundials-{}.tar.gz".format(sundials_version)
    )
    logger.info("Downloading sundials")
    download_extract_library(url, directory)

    cmake_args = [
        "-DLAPACK_ENABLE=ON",
        "-DSUNDIALS_INDEX_SIZE=32",
        "-DBUILD_ARKODE:BOOL=OFF",
        "-DEXAMPLES_ENABLE:BOOL=OFF",
        "-DCMAKE_INSTALL_PREFIX=" + join(directory, "sundials5"),
    ]

    # SUNDIALS are built within directory 'build_sundials' in the PyBaMM root
    # directory
    build_directory = os.path.abspath(join(directory, "build_sundials"))
    if not os.path.exists(build_directory):
        print("\n-" * 10, "Creating build dir", "-" * 40)
        os.makedirs(build_directory)

    print("-" * 10, "Running CMake prepare", "-" * 40)
    subprocess.run(
        ["cmake", "../sundials-{}".format(sundials_version)] + cmake_args,
        cwd=build_directory,
    )

    print("-" * 10, "Building the sundials", "-" * 40)
    make_cmd = ["make", "install"]
    subprocess.run(make_cmd, cwd=build_directory)


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


def main(arguments=None):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("scikits.odes setup")

    # To override the default severity of logging
    logger.setLevel("INFO")

    # Use FileHandler() to log to a file
    logfile = join(os.path.dirname(os.path.abspath(__file__)), "scikits_odes_setup.log")
    print(logfile)
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    # Add the file handler
    logger.addHandler(file_handler)
    logger.info("Starting scikits.odes setup")

    desc = "Install scikits.odes."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--sundials-libs", type=str, help="path to sundials libraries.")
    parser.add_argument("--install-sundials", action="store_true")
    args = parser.parse_args(arguments)

    if args.install_sundials:
        logger.info("Installing sundials")
        install_sundials()

    # Check is sundials is already installed
    SUNDIALS_LIB_DIRS = [
        join(pybamm_dir(), "scikits.odes/sundials5"),
        join(os.getenv("HOME"), ".local"),
        "/usr/local",
        "/usr",
    ]
    if args.sundials_libs:
        SUNDIALS_LIB_DIRS.insert(0, args.sundials_libs)
    for DIR in SUNDIALS_LIB_DIRS:
        logger.info("Looking for sundials at {}".format(DIR))
        SUNDIALS_FOUND = isfile(join(DIR, "lib", "libsundials_ida.so")) or isfile(
            join(DIR, "lib", "libsundials_ida.dylib")
        )
        SUNDIALS_LIB_DIR = DIR if SUNDIALS_FOUND else ""
        if SUNDIALS_FOUND:
            logger.info("Found sundials at {}".format(SUNDIALS_LIB_DIR))
            break

    if not SUNDIALS_FOUND:
        raise RuntimeError("Could not find sundials libraries.")

    update_LD_LIBRARY_PATH(SUNDIALS_LIB_DIR)

    # At the time scikits.odes is pip installed, the path to the sundials
    # library must be contained in an env variable SUNDIALS_INST
    # see https://scikits-odes.readthedocs.io/en/latest/installation.html#id1
    os.environ["SUNDIALS_INST"] = SUNDIALS_LIB_DIR
    env = os.environ.copy()
    subprocess.run(["pip", "install", "scikits.odes"], env=env)


if __name__ == "__main__":
    main(sys.argv[1:])
