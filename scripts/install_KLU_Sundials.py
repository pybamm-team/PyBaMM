import os
import subprocess
import tarfile
import argparse
import platform
import hashlib
import shutil
import urllib.request
from os.path import join, isfile
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count


SUITESPARSE_VERSION = "6.0.3"
SUNDIALS_VERSION = "6.5.0"

SUITESPARSE_URL = f"https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v{SUITESPARSE_VERSION}.tar.gz"
SUNDIALS_URL = f"https://github.com/LLNL/sundials/releases/download/v{SUNDIALS_VERSION}/sundials-{SUNDIALS_VERSION}.tar.gz"

SUITESPARSE_CHECKSUM = (
    "7111b505c1207f6f4bd0be9740d0b2897e1146b845d73787df07901b4f5c1fb7"
)
SUNDIALS_CHECKSUM = "4e0b998dff292a2617e179609b539b511eb80836f5faacf800e688a886288502"

# universal binaries for macOS 11.0 and later; sourced from https://mac.r-project.org/openmp/
OPENMP_VERSION = "16.0.4"
OPENMP_URL = (
    f"https://mac.r-project.org/openmp/openmp-{OPENMP_VERSION}-darwin20-Release.tar.gz"
)
OPENMP_CHECKSUM = "a763f0bdc9115c4f4933accc81f514f3087d56d6528778f38419c2a0d2231972"


DEFAULT_INSTALL_DIR = os.path.join(os.getenv("HOME"), ".local")


def safe_remove_dir(path):
    """Remove a directory or file if it exists."""
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"Error while removing {path}: {e}")


def install_suitesparse(download_dir):
    # The SuiteSparse KLU module has 4 dependencies:
    # - suitesparseconfig
    # - AMD
    # - COLAMD
    # - BTF
    suitesparse_dir = f"SuiteSparse-{SUITESPARSE_VERSION}"
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
        # Set an RPATH in order for libsuitesparseconfig.dylib to find libomp.dylib
        if libdir == "SuiteSparse_config":
            env["CMAKE_OPTIONS"] = (
                f"-DCMAKE_INSTALL_PREFIX={install_dir} -DCMAKE_INSTALL_RPATH={install_dir}/lib"
            )
        else:
            # For AMD, COLAMD, BTF and KLU; do not set a BUILD RPATH but use an
            # INSTALL RPATH in order to ensure that the dynamic libraries are found
            # at runtime just once. Otherwise, delocate complains about multiple
            # references to the SuiteSparse_config dynamic library (auditwheel does not).
            env["CMAKE_OPTIONS"] = (
                f"-DCMAKE_INSTALL_PREFIX={install_dir} -DCMAKE_INSTALL_RPATH={install_dir}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE -DCMAKE_BUILD_WITH_INSTALL_RPATH=FALSE"
            )
        subprocess.run(make_cmd, cwd=build_dir, env=env, shell=True, check=True)
        subprocess.run(install_cmd, cwd=build_dir, check=True)


def install_sundials(download_dir, install_dir):
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
        OpenMP_C_FLAGS = f"-Xpreprocessor -fopenmp -lomp -L{os.path.join(KLU_LIBRARY_DIR)} -I{os.path.join(KLU_INCLUDE_DIR)}"
        OpenMP_C_LIB_NAMES = "omp"
        OpenMP_omp_LIBRARY = os.path.join(KLU_LIBRARY_DIR, "libomp.dylib")

        cmake_args += [
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

    sundials_src = f"../sundials-{SUNDIALS_VERSION}"
    print("-" * 10, "Running CMake prepare", "-" * 40)
    subprocess.run(["cmake", sundials_src, *cmake_args], cwd=build_dir, check=True)

    print("-" * 10, "Building SUNDIALS", "-" * 40)
    make_cmd = ["make", f"-j{cpu_count()}", "install"]
    subprocess.run(make_cmd, cwd=build_dir, check=True)


# Relevant for macOS only because recent Xcode Clang versions do not include OpenMP headers.
# Other compilers (e.g. GCC) include the OpenMP specification by default.
def set_up_openmp(download_dir, install_dir):
    print("-" * 10, "Extracting OpenMP archive", "-" * 40)

    openmp_dir = f"openmp-{OPENMP_VERSION}"
    openmp_src = os.path.join(download_dir, openmp_dir)

    # extract OpenMP archive
    with tarfile.open(
        os.path.join(download_dir, f"{openmp_dir}-darwin20-Release.tar.gz")
    ) as tar:
        tar.extractall(openmp_src)

    # create directories
    os.makedirs(os.path.join(install_dir, "lib"), exist_ok=True)
    os.makedirs(os.path.join(install_dir, "include"), exist_ok=True)

    # copy files
    shutil.copy(
        os.path.join(openmp_src, "usr", "local", "lib", "libomp.dylib"),
        os.path.join(install_dir, "lib"),
    )
    for file in os.listdir(os.path.join(openmp_src, "usr", "local", "include")):
        shutil.copy(
            os.path.join(openmp_src, "usr", "local", "include", file),
            os.path.join(install_dir, "include"),
        )

    # fix rpath; for some reason the downloaded dylib has an absolute path
    # to /usr/local/lib/, so use self-referential rpath
    subprocess.check_call(
        [
            "install_name_tool",
            "-id",
            "@rpath/libomp.dylib",
            f"{os.path.join(install_dir, 'lib', 'libomp.dylib')}",
        ]
    )


def check_libraries_installed(install_dir):
    # Define the directories to check for SUNDIALS and SuiteSparse libraries
    lib_dirs = [install_dir]

    sundials_files = [
        "libsundials_idas",
        "libsundials_sunlinsolklu",
        "libsundials_sunlinsoldense",
        "libsundials_sunlinsolspbcgs",
        "libsundials_sunlinsollapackdense",
        "libsundials_sunmatrixsparse",
        "libsundials_nvecserial",
        "libsundials_nvecopenmp",
    ]
    if platform.system() == "Linux":
        sundials_files = [file + ".so" for file in sundials_files]
    elif platform.system() == "Darwin":
        sundials_files = [file + ".dylib" for file in sundials_files]
    sundials_lib_found = True
    # Check for SUNDIALS libraries in each directory
    for lib_file in sundials_files:
        file_found = False
        for lib_dir in lib_dirs:
            if isfile(join(lib_dir, "lib", lib_file)):
                print(f"{lib_file} found in {lib_dir}.")
                file_found = True
                break
        if not file_found:
            print(
                f"{lib_file} not found. Proceeding with SUNDIALS library installation."
            )
            sundials_lib_found = False
            break

    suitesparse_files = [
        "libsuitesparseconfig",
        "libklu",
        "libamd",
        "libcolamd",
        "libbtf",
    ]
    if platform.system() == "Linux":
        suitesparse_files = [file + ".so" for file in suitesparse_files]
    elif platform.system() == "Darwin":
        suitesparse_files = [file + ".dylib" for file in suitesparse_files]
    else:
        raise NotImplementedError(
            f"Unsupported operating system: {platform.system()}. This script supports only Linux and macOS."
        )

    suitesparse_lib_found = True
    # Check for SuiteSparse libraries in each directory
    for lib_file in suitesparse_files:
        file_found = False
        for lib_dir in lib_dirs:
            if isfile(join(lib_dir, "lib", lib_file)):
                print(f"{lib_file} found in {lib_dir}.")
                file_found = True
                break
        if not file_found:
            print(
                f"{lib_file} not found. Proceeding with SuiteSparse library installation."
            )
            suitesparse_lib_found = False
            break

    return sundials_lib_found, suitesparse_lib_found


def check_openmp_installed_on_macos(install_dir):
    openmp_lib_found = isfile(join(install_dir, "lib", "libomp.dylib"))
    openmp_headers_found = isfile(join(install_dir, "include", "omp.h"))
    if not openmp_lib_found or not openmp_headers_found:
        print("libomp.dylib or omp.h not found. Proceeding with OpenMP installation.")
    else:
        print(f"libomp.dylib and omp.h found in {install_dir}.")
    return openmp_lib_found


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_extract_library(url, expected_checksum, download_dir):
    file_name = url.split("/")[-1]
    file_path = os.path.join(download_dir, file_name)

    # Check if file already exists and validate checksum
    if os.path.exists(file_path):
        print(f"Validating checksum for {file_name}...")
        actual_checksum = calculate_sha256(file_path)
        print(f"Found {actual_checksum} against {expected_checksum}")
        if actual_checksum == expected_checksum:
            print(f"Checksum valid. Skipping download for {file_name}.")
            # Extract the archive as the checksum is valid
            with tarfile.open(file_path) as tar:
                tar.extractall(download_dir)
            return
        else:
            print(f"Checksum invalid. Redownloading {file_name}.")

    # Download and extract archive at url
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(
            f"Invalid URL scheme: {parsed_url.scheme}. Only HTTP and HTTPS are allowed."
        )
    with urllib.request.urlopen(url) as response:
        os.makedirs(download_dir, exist_ok=True)
        with open(file_path, "wb") as out_file:
            out_file.write(response.read())
    with tarfile.open(file_path) as tar:
        tar.extractall(download_dir)


def parallel_download(urls, download_dir):
    # Use 2 processes for parallel downloading
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = [
            executor.submit(
                download_extract_library, url, expected_checksum, download_dir
            )
            for (url, expected_checksum) in urls
        ]
        for future in futures:
            future.result()


# First check requirements: make and cmake
try:
    subprocess.run(["make", "--version"])
except OSError as error:
    raise RuntimeError("Make must be installed.") from error
try:
    subprocess.run(["cmake", "--version"])
except OSError as error:
    raise RuntimeError("CMake must be installed.") from error

# Build in parallel wherever possible
os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())

# Create download directory in PyBaMM dir
pybamm_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
download_dir = os.path.join(pybamm_dir, "install_KLU_Sundials")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Get installation location
parser = argparse.ArgumentParser(
    description="Download, compile and install SUNDIALS and SuiteSparse."
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Force installation even if libraries are already found. This will overwrite the pre-existing files.",
)
parser.add_argument("--install-dir", type=str, default=DEFAULT_INSTALL_DIR)
args = parser.parse_args()
install_dir = (
    args.install_dir
    if os.path.isabs(args.install_dir)
    else os.path.join(pybamm_dir, args.install_dir)
)

if args.force:
    print(
        "The '--force' option is activated: installation will be forced, ignoring any existing libraries."
    )
    safe_remove_dir(os.path.join(download_dir, "build_sundials"))
    safe_remove_dir(os.path.join(download_dir, f"SuiteSparse-{SUITESPARSE_VERSION}"))
    safe_remove_dir(os.path.join(download_dir, f"sundials-{SUNDIALS_VERSION}"))
    if platform.system() == "Darwin":
        safe_remove_dir(os.path.join(install_dir, "lib", "libomp.dylib"))
        safe_remove_dir(os.path.join(install_dir, "include", "omp.h"))
        sundials_found, suitesparse_found, openmp_found = False, False, False
    else:
        sundials_found, suitesparse_found = False, False
else:
    # Check whether the libraries are installed
    if platform.system() == "Darwin":
        sundials_found, suitesparse_found = check_libraries_installed(install_dir)
        openmp_found = check_openmp_installed_on_macos(install_dir)
    else:  # Linux
        sundials_found, suitesparse_found = check_libraries_installed(install_dir)


if __name__ == "__main__":
    # Determine which libraries to download based on whether they were found
    if not sundials_found and not suitesparse_found:
        # Both SUNDIALS and SuiteSparse are missing, download and install both
        parallel_download(
            [
                (SUITESPARSE_URL, SUITESPARSE_CHECKSUM),
                (SUNDIALS_URL, SUNDIALS_CHECKSUM),
            ],
            download_dir,
        )

        if platform.system() == "Darwin" and not openmp_found:
            download_extract_library(OPENMP_URL, OPENMP_CHECKSUM, download_dir)
            set_up_openmp(download_dir, install_dir)

        install_suitesparse(download_dir)
        install_sundials(download_dir, install_dir)

    else:
        if not sundials_found:
            # Only SUNDIALS is missing, download and install it
            parallel_download([(SUNDIALS_URL, SUNDIALS_CHECKSUM)], download_dir)
            if platform.system() == "Darwin" and not openmp_found:
                download_extract_library(OPENMP_URL, OPENMP_CHECKSUM, download_dir)
                set_up_openmp(download_dir, install_dir)
            # openmp needed for SUNDIALS on macOS
            install_sundials(download_dir, install_dir)
        if not suitesparse_found:
            # Only SuiteSparse is missing, download and install it
            parallel_download([(SUITESPARSE_URL, SUITESPARSE_CHECKSUM)], download_dir)
            install_suitesparse(download_dir)
