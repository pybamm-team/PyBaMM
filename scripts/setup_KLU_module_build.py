import os
import subprocess
import tarfile

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


# First check requirements: make and cmake
try:
    subprocess.run(["make", "--version"])
except OSError:
    raise RuntimeError("Make must be installed.")
try:
    subprocess.run(["cmake", "--version"])
except OSError:
    raise RuntimeError("CMake must be installed.")

# Get abs path to dir containing "scripts/"
# Likely PyBaMM/
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
directory = os.path.join(parent_dir, "KLU_module_deps")
if not os.path.exists(directory):
    os.makedirs(directory)

# Download SuiteSparse
suitesparse_url = (
    "https://github.com/DrTimothyAldenDavis/" + "SuiteSparse/archive/v5.6.0.tar.gz"
)
download_extract_library(suitesparse_url, directory)

# The SuiteSparse KLU module has 4 dependencies:
# - suitesparseconfig
# - AMD
# - COLAMD
# - BTF
suitesparse_src = os.path.join(directory, "SuiteSparse-5.6.0")
print("-" * 10, "Building SuiteSparse_config", "-" * 40)
make_cmd = ["make"]
build_dir = os.path.join(suitesparse_src, "SuiteSparse_config")
subprocess.run(make_cmd, cwd=build_dir)

print("-" * 10, "Building SuiteSparse KLU module dependencies", "-" * 40)
make_cmd = ["make", "library"]
for libdir in ["AMD", "COLAMD", "BTF"]:
    build_dir = os.path.join(suitesparse_src, libdir)
    subprocess.run(make_cmd, cwd=build_dir)

print("-" * 10, "Building SuiteSparse KLU module", "-" * 40)
build_dir = os.path.join(suitesparse_src, "KLU")
subprocess.run(make_cmd, cwd=build_dir)

# Download SUNDIALS
sundials_url = (
    "https://computing.llnl.gov/" + "projects/sundials/download/sundials-5.0.0.tar.gz"
)
download_extract_library(sundials_url, directory)

sundials_inst = "../sundials5"
cmake_args = [
    "-DLAPACK_ENABLE=ON",
    "-DSUNDIALS_INDEX_SIZE=32",
    "-DBUILD_ARKODE:BOOL=OFF",
    "-DBUILD_CVODE=OFF",
    "-DBUILD_CVODES=OFF",
    "-DBUILD_IDAS=OFF",
    "-DBUILD_KINSOL=OFF",
    "-DEXAMPLES_ENABLE:BOOL=OFF",
    "-DKLU_ENABLE=ON",
    "-DKLU_INCLUDE_DIR=../SuiteSparse-5.6.0/include",
    "-DKLU_LIBRARY_DIR=../SuiteSparse-5.6.0/lib",
    "-DCMAKE_INSTALL_PREFIX=" + sundials_inst,
]

# SUNDIALS are built within directory 'build_sundials' in the PyBaMM root
# directory
build_directory = os.path.abspath(os.path.join(directory, "build_sundials"))
if not os.path.exists(build_directory):
    print("\n-" * 10, "Creating build dir", "-" * 40)
    os.makedirs(build_directory)

sundials_src = "../sundials-5.0.0"
print("-" * 10, "Running CMake prepare", "-" * 40)
subprocess.run(["cmake", sundials_src] + cmake_args, cwd=build_directory)

print("-" * 10, "Building the sundials", "-" * 40)
make_cmd = ["make", "install"]
subprocess.run(make_cmd, cwd=build_directory)
