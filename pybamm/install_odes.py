import os
import tarfile
from os.path import join, isfile
import argparse
import sys
import logging
import subprocess

from pybamm.util import root_dir

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


def install_sundials(download_dir, install_dir):
    # Download the SUNDIALS library and compile it.
    logger = logging.getLogger("scikits.odes setup")
    sundials_version = "6.5.0"

    try:
        subprocess.run(["cmake", "--version"])
    except OSError:
        raise RuntimeError("CMake must be installed to build SUNDIALS.")

    url = (
        "https://github.com/LLNL/"
        + "sundials/releases/download/v{}/sundials-{}.tar.gz".format(
            sundials_version, sundials_version
        )
    )
    logger.info("Downloading sundials")
    download_extract_library(url, download_dir)

    cmake_args = [
        "-DLAPACK_ENABLE=ON",
        "-DSUNDIALS_INDEX_SIZE=32",
        "-DBUILD_ARKODE:BOOL=OFF",
        "-DEXAMPLES_ENABLE:BOOL=OFF",
        "-DCMAKE_INSTALL_PREFIX=" + install_dir,
    ]

    # SUNDIALS are built within directory 'build_sundials' in the PyBaMM root
    # directory
    build_directory = os.path.abspath(join(download_dir, "build_sundials"))
    if not os.path.exists(build_directory):
        print("\n-" * 10, "Creating build dir", "-" * 40)
        os.makedirs(build_directory)

    print("-" * 10, "Running CMake prepare", "-" * 40)
    subprocess.run(
        ["cmake", f"../sundials-{sundials_version}", *cmake_args],
        cwd=build_directory,
        check=True,
    )

    print("-" * 10, "Building the sundials", "-" * 40)
    make_cmd = ["make", "install"]
    subprocess.run(make_cmd, cwd=build_directory, check=True)


def update_LD_LIBRARY_PATH(install_dir):
    # Look for current python virtual env and add export statement
    # for LD_LIBRARY_PATH in activate script.  If no virtual env found,
    # then the current user's .bashrc file is modified instead.

    export_statement = f"export LD_LIBRARY_PATH={install_dir}/lib:$LD_LIBRARY_PATH"

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        script_path = os.path.join(venv_path, "bin/activate")
    else:
        script_path = os.path.join(os.environ.get("HOME"), ".bashrc")

    if os.getenv("LD_LIBRARY_PATH") and f"{install_dir}/lib" in os.getenv(
        "LD_LIBRARY_PATH"
    ):
        print(f"{install_dir}/lib was found in LD_LIBRARY_PATH.")
        print("--> Not updating venv activate or .bashrc scripts")
    else:
        with open(script_path, "a+") as fh:
            # Just check that export statement is not already there.
            if export_statement not in fh.read():
                fh.write(export_statement)
                print(
                    f"Adding {install_dir}/lib to LD_LIBRARY_PATH"
                    f" in {script_path}"
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
    default_install_dir = os.path.join(os.getenv("HOME"), ".local")
    parser.add_argument("--install-dir", type=str, default=default_install_dir)
    args = parser.parse_args(arguments)

    pybamm_dir = root_dir()
    install_dir = (
        args.install_dir
        if os.path.isabs(args.install_dir)
        else os.path.join(pybamm_dir, args.install_dir)
    )

    # Check is sundials is already installed
    SUNDIALS_LIB_DIRS = [join(os.getenv("HOME"), ".local"), "/usr/local", "/usr"]

    if args.sundials_libs:
        SUNDIALS_LIB_DIRS.insert(0, args.sundials_libs)
    for DIR in SUNDIALS_LIB_DIRS:
        logger.info(f"Looking for sundials at {DIR}")
        SUNDIALS_FOUND = isfile(join(DIR, "lib", "libsundials_ida.so")) or isfile(
            join(DIR, "lib", "libsundials_ida.dylib")
        )
        if SUNDIALS_FOUND:
            SUNDIALS_LIB_DIR = DIR
            logger.info(f"Found sundials at {SUNDIALS_LIB_DIR}")
            break

    if not SUNDIALS_FOUND:
        logger.info("Could not find sundials libraries.")
        logger.info(f"Installing sundials in {install_dir}")
        download_dir = os.path.join(pybamm_dir, "sundials")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        install_sundials(download_dir, install_dir)
        SUNDIALS_LIB_DIR = install_dir

    update_LD_LIBRARY_PATH(SUNDIALS_LIB_DIR)

    # At the time scikits.odes is pip installed, the path to the sundials
    # library must be contained in an env variable SUNDIALS_INST
    # see https://scikits-odes.readthedocs.io/en/latest/installation.html#id1
    os.environ["SUNDIALS_INST"] = SUNDIALS_LIB_DIR
    env = os.environ.copy()
    subprocess.run(["pip", "install", "scikits.odes"], env=env, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
