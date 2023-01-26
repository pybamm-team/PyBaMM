import os
import subprocess
import tarfile
import argparse

try:
    # wget module is required to download SUNDIALS or SuiteSparse.
    import wget

    NO_WGET = False
except ModuleNotFoundError:
    NO_WGET = True


def download_extract_library(url, download_dir):
    # Download and extract archive at url
    if NO_WGET:
        error_msg = (
            "Could not find wget module."
            " Please install wget module (pip install wget)."
        )
        raise ModuleNotFoundError(error_msg)
    archive = wget.download(url, out=download_dir)
    tar = tarfile.open(archive)
    tar.extractall(download_dir)


# First check requirements: make and cmake
try:
    subprocess.run(["make", "--version"])
except OSError:
    raise RuntimeError("Make must be installed.")
try:
    subprocess.run(["cmake", "--version"])
except OSError:
    raise RuntimeError("CMake must be installed.")

# Create download directory in PyBaMM dir
pybamm_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
download_dir = os.path.join(pybamm_dir, "install_KLU_Sundials")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Get installation location
default_install_dir = os.path.join(os.getenv("HOME"), ".local")
parser = argparse.ArgumentParser(
    description="Download, compile and install Sundials and SuiteSparse."
)
parser.add_argument("--install-dir", type=str, default=default_install_dir)
args = parser.parse_args()
install_dir = (
    args.install_dir
    if os.path.isabs(args.install_dir)
    else os.path.join(pybamm_dir, args.install_dir)
)

# 1 --- Download SuiteSparse
suitesparse_version = "6.0.3"
suitesparse_url = (
    "https://github.com/DrTimothyAldenDavis/"
    + "SuiteSparse/archive/v{}.tar.gz".format(suitesparse_version)
)
download_extract_library(suitesparse_url, download_dir)

# The SuiteSparse KLU module has 4 dependencies:
# - suitesparseconfig
# - AMD
# - COLAMD
# - BTF
suitesparse_dir = "SuiteSparse-{}".format(suitesparse_version)
suitesparse_src = os.path.join(download_dir, suitesparse_dir)
print("-" * 10, "Building SuiteSparse_config", "-" * 40)
make_cmd = [
    "make",
    "library",
    'CMAKE_OPTIONS="-DCMAKE_INSTALL_PREFIX={}"'.format(install_dir),
]
install_cmd = [
    "make",
    "install",
]
print("-" * 10, "Building SuiteSparse", "-" * 40)
for libdir in ["SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU"]:
    build_dir = os.path.join(suitesparse_src, libdir)
    subprocess.run(make_cmd, cwd=build_dir)
    subprocess.run(install_cmd, cwd=build_dir)

# 2 --- Download SUNDIALS
sundials_version = "6.5.0"
sundials_url = (
    "https://github.com/LLNL/sundials/"
    + f"releases/download/v{sundials_version}/sundials-{sundials_version}.tar.gz"
)

download_extract_library(sundials_url, download_dir)

# Set install dir for SuiteSparse libs
# Ex: if install_dir -> "/usr/local/" then
# KLU_INCLUDE_DIR -> "/usr/local/include"
# KLU_LIBRARY_DIR -> "/usr/local/lib"
KLU_INCLUDE_DIR = os.path.join(install_dir, "include")
KLU_LIBRARY_DIR = os.path.join(install_dir, "lib")
cmake_args = [
    "-DLAPACK_ENABLE=ON",
    "-DSUNDIALS_INDEX_SIZE=32",
    "-DEXAMPLES_ENABLE:BOOL=OFF",
    "-DKLU_ENABLE=ON",
    "-DKLU_INCLUDE_DIR={}".format(KLU_INCLUDE_DIR),
    "-DKLU_LIBRARY_DIR={}".format(KLU_LIBRARY_DIR),
    "-DCMAKE_INSTALL_PREFIX=" + install_dir,
    # on mac use fixed paths rather than rpath
    "-DCMAKE_INSTALL_NAME_DIR=" + KLU_LIBRARY_DIR,
]

# SUNDIALS are built within download_dir 'build_sundials' in the PyBaMM root
# download_dir
build_dir = os.path.abspath(os.path.join(download_dir, "build_sundials"))
if not os.path.exists(build_dir):
    print("\n-" * 10, "Creating build dir", "-" * 40)
    os.makedirs(build_dir)

sundials_src = "../sundials-{}".format(sundials_version)
print("-" * 10, "Running CMake prepare", "-" * 40)
subprocess.run(["cmake", sundials_src] + cmake_args, cwd=build_dir)

print("-" * 10, "Building the sundials", "-" * 40)
make_cmd = ["make", "install"]
subprocess.run(make_cmd, cwd=build_dir)
