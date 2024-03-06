import os
import tarfile
from os.path import join, isfile
import argparse
import sys
import logging
import subprocess
from multiprocessing import cpu_count

from pybamm.util import root_dir

if sys.platform == "win32":
    raise Exception("pybamm_install_odes is not supported on Windows.")

SUNDIALS_VERSION = "6.5.0"

# Build in parallel wherever possible
os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())

try:
    # wget module is required to download SUNDIALS or SuiteSparse.
    import wget

    NO_WGET = False
except ModuleNotFoundError:
    NO_WGET = True

# Build in parallel wherever possible
os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())


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

    try:
        subprocess.run(["cmake", "--version"])
    except OSError as error:
        raise RuntimeError("CMake must be installed to build SUNDIALS.") from error

    url = f"https://github.com/LLNL/sundials/releases/download/v{SUNDIALS_VERSION}/sundials-{SUNDIALS_VERSION}.tar.gz"
    logger.info("Downloading sundials")
    download_extract_library(url, download_dir)

    cmake_args = [
        "-DLAPACK_ENABLE=ON",
        "-DSUNDIALS_INDEX_SIZE=32",
        "-DBUILD_ARKODE:BOOL=OFF",
        "-DEXAMPLES_ENABLE:BOOL=OFF",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
    ]

    # SUNDIALS are built within directory 'build_sundials' in the PyBaMM root
    # directory
    build_directory = os.path.abspath(join(download_dir, "build_sundials"))
    if not os.path.exists(build_directory):
        print("\n-" * 10, "Creating build dir", "-" * 40)
        os.makedirs(build_directory)

    print("-" * 10, "Running CMake prepare", "-" * 40)
    subprocess.run(
        ["cmake", f"../sundials-{SUNDIALS_VERSION}", *cmake_args],
        cwd=build_directory,
        check=True,
    )

    print("-" * 10, "Building the sundials", "-" * 40)
    make_cmd = ["make", "install"]
    subprocess.run(make_cmd, cwd=build_directory, check=True)


def update_LD_LIBRARY_PATH(install_dir):
    # Look for the current python virtual env and add an export statement
    # for LD_LIBRARY_PATH in the activate script. If no virtual env is found,
    # the current user's .bashrc file is modified instead.

    export_statement = f"export LD_LIBRARY_PATH={install_dir}/lib:$LD_LIBRARY_PATH"

    home_dir = os.environ.get("HOME")
    bashrc_path = os.path.join(home_dir, ".bashrc")
    zshrc_path = os.path.join(home_dir, ".zshrc")
    venv_path = os.environ.get("VIRTUAL_ENV")

    if venv_path:
        script_path = os.path.join(venv_path, "bin/activate")
    else:
        if os.path.exists(bashrc_path):
            script_path = os.path.join(os.environ.get("HOME"), ".bashrc")
        elif os.path.exists(zshrc_path):
            script_path = os.path.join(os.environ.get("HOME"), ".zshrc")
        elif os.path.exists(bashrc_path) and os.path.exists(zshrc_path):
            print(
                "Both .bashrc and .zshrc found in the home directory. Setting .bashrc as path"
            )
            script_path = os.path.join(os.environ.get("HOME"), ".bashrc")
        else:
            print("Neither .bashrc nor .zshrc found in the home directory.")

    if os.getenv("LD_LIBRARY_PATH") and f"{install_dir}/lib" in os.getenv(
        "LD_LIBRARY_PATH"
    ):
        print(f"{install_dir}/lib was found in LD_LIBRARY_PATH.")
        if os.path.exists(bashrc_path):
            print("--> Not updating venv activate or .bashrc scripts")
        if os.path.exists(zshrc_path):
            print("--> Not updating venv activate or .zshrc scripts")
    else:
        with open(script_path, "a+") as fh:
            # Just check that export statement is not already there.
            if export_statement not in fh.read():
                fh.write(export_statement)
                print(
                    f"Adding {install_dir}/lib to LD_LIBRARY_PATH" f" in {script_path}"
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

    # Check if sundials is already installed
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
    logger.info("Installing scikits.odes via pip")
    logger.info("Purging scikits.odes whels from pip cache if present")
    subprocess.run(
        [f"{sys.executable}", "-m", "pip", "cache", "remove", "scikits.odes"],
        check=True,
    )
    subprocess.run(
        [f"{sys.executable}", "-m", "pip", "install", "scikits.odes", "--verbose"],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
