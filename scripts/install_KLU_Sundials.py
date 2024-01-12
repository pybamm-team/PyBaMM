# /// pyproject
# [run]
# requires-python = "">=3.8, <3.13""
# dependencies = [
#   "cmake",
# ]
#
# [additional-info]
# repository = "https://github.com/pybamm-team/PyBaMM"
# documentation = "https://docs.pybamm.org"
# ///
import os
import subprocess
import tarfile
import argparse
import platform
import concurrent.futures
import urllib.request
from multiprocessing import cpu_count


def download_extract_library(url, download_dir):
    # Download and extract archive at url
    file_name = url.split("/")[-1]
    file_path = os.path.join(download_dir, file_name)
    with urllib.request.urlopen(url) as response:
        os.makedirs(download_dir, exist_ok=True)
        with open(file_path, "wb") as out_file:
            out_file.write(response.read())
    with tarfile.open(file_path) as tar:
        tar.extractall(download_dir)


def parallel_download(urls, download_dir):
    # Use 2 processes for parallel downloading
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_extract_library, url, download_dir) for url in urls
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


# First check requirements: make and cmake
try:
    subprocess.run(["make", "--version"])
except OSError:
    raise RuntimeError("Make must be installed.")
try:
    subprocess.run(["cmake", "--version"])
except OSError:
    raise RuntimeError("CMake must be installed.")

# Build in parallel wherever possible
os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())

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

# Parallel download

# 1 --- SuiteSparse
suitesparse_version = "6.0.3"
suitesparse_url = (
    "https://github.com/DrTimothyAldenDavis/"
    + f"SuiteSparse/archive/v{suitesparse_version}.tar.gz"
)

# 2 --- SUNDIALS
sundials_version = "6.5.0"
sundials_url = (
    "https://github.com/LLNL/sundials/"
    + f"releases/download/v{sundials_version}/sundials-{sundials_version}.tar.gz"
)

parallel_download([suitesparse_url, sundials_url], download_dir)

# 1 --- Install SuiteSparse

# The SuiteSparse KLU module has 4 dependencies:
# - suitesparseconfig
# - AMD
# - COLAMD
# - BTF
suitesparse_dir = f"SuiteSparse-{suitesparse_version}"
suitesparse_src = os.path.join(download_dir, suitesparse_dir)
print("-" * 10, "Building SuiteSparse_config", "-" * 40)
make_cmd = [
    "make",
    "library",
]
install_cmd = [
    "make",
    f"-j{cpu_count()}",
    "install",
]
print("-" * 10, "Building SuiteSparse", "-" * 40)
# Set CMAKE_OPTIONS as environment variables to pass to the GNU Make command
env = os.environ.copy()
for libdir in ["SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU"]:
    build_dir = os.path.join(suitesparse_src, libdir)
    # We want to ensure that libsuitesparseconfig.dylib is not repeated in
    # multiple paths at the time of wheel repair. Therefore, it should not be
    # built with an RPATH since it is copied to the install prefix.
    if libdir == "SuiteSparse_config":
        env["CMAKE_OPTIONS"] = f"-DCMAKE_INSTALL_PREFIX={install_dir}"
    else:
        # For AMD, COLAMD, BTF and KLU; do not set a BUILD RPATH but use an
        # INSTALL RPATH in order to ensure that the dynamic libraries are found
        # at runtime just once. Otherwise, delocate complains about multiple
        # references to the SuiteSparse_config dynamic library (auditwheel does not).
        env[
            "CMAKE_OPTIONS"
        ] = f"-DCMAKE_INSTALL_PREFIX={install_dir} -DCMAKE_INSTALL_RPATH={install_dir}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE -DCMAKE_BUILD_WITH_INSTALL_RPATH=FALSE"
    subprocess.run(make_cmd, cwd=build_dir, env=env, shell=True, check=True)
    subprocess.run(install_cmd, cwd=build_dir, check=True)

# 2 --- Install SUNDIALS

# Set install dir for SuiteSparse libs
# Ex: if install_dir -> "/usr/local/" then
# KLU_INCLUDE_DIR -> "/usr/local/include"
# KLU_LIBRARY_DIR -> "/usr/local/lib"
KLU_INCLUDE_DIR = os.path.join(install_dir, "include")
KLU_LIBRARY_DIR = os.path.join(install_dir, "lib")
cmake_args = [
    "-DENABLE_LAPACK=ON",
    "-DSUNDIALS_INDEX_SIZE=32",
    "-DEXAMPLES_ENABLE_C=OFF",
    "-DEXAMPLES_ENABLE_CXX=OFF",
    "-DEXAMPLES_INSTALL=OFF",
    "-DENABLE_KLU=ON",
    "-DENABLE_OPENMP=ON",
    f"-DKLU_INCLUDE_DIR={KLU_INCLUDE_DIR}",
    f"-DKLU_LIBRARY_DIR={KLU_LIBRARY_DIR}",
    "-DCMAKE_INSTALL_PREFIX=" + install_dir,
    # on macOS use fixed paths rather than rpath
    "-DCMAKE_INSTALL_NAME_DIR=" + KLU_LIBRARY_DIR,
]

# try to find OpenMP on mac
if platform.system() == "Darwin":
    # flags to find OpenMP on mac
    if platform.processor() == "arm":
        LDFLAGS = "-L/opt/homebrew/opt/libomp/lib"
        CPPFLAGS = "-I/opt/homebrew/opt/libomp/include"
        OpenMP_C_FLAGS = "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        OpenMP_C_LIB_NAMES = "omp"
        OpenMP_libomp_LIBRARY = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
        OpenMP_omp_LIBRARY = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    elif platform.processor() == "i386":
        LDFLAGS = "-L/usr/local/opt/libomp/lib"
        CPPFLAGS = "-I/usr/local/opt/libomp/include"
        OpenMP_C_FLAGS = "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
        OpenMP_CXX_FLAGS = "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
        OpenMP_C_LIB_NAMES = "omp"
        OpenMP_CXX_LIB_NAMES = "omp"
        OpenMP_omp_LIBRARY = "/usr/local/opt/libomp/lib/libomp.dylib"

    cmake_args += [
        "-DLDFLAGS=" + LDFLAGS,
        "-DCPPFLAGS=" + CPPFLAGS,
        "-DOpenMP_C_FLAGS=" + OpenMP_C_FLAGS,
        "-DOpenMP_C_LIB_NAMES=" + OpenMP_C_LIB_NAMES,
        "-DOpenMP_omp_LIBRARY=" + OpenMP_omp_LIBRARY,
    ]

# SUNDIALS are built within download_dir 'build_sundials' in the PyBaMM root
# download_dir
build_dir = os.path.abspath(os.path.join(download_dir, "build_sundials"))
if not os.path.exists(build_dir):
    print("\n-" * 10, "Creating build dir", "-" * 40)
    os.makedirs(build_dir)

sundials_src = f"../sundials-{sundials_version}"
print("-" * 10, "Running CMake prepare", "-" * 40)
subprocess.run(["cmake", sundials_src, *cmake_args], cwd=build_dir, check=True)

print("-" * 10, "Building the sundials", "-" * 40)
make_cmd = ["make", f"-j{cpu_count()}", "install"]
subprocess.run(make_cmd, cwd=build_dir, check=True)
