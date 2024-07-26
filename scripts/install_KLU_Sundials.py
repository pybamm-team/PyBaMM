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
DEFAULT_INSTALL_DIR = os.path.join(os.getenv("HOME"), ".local")


def safe_remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


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
        # We want to ensure that libsuitesparseconfig.dylib is not repeated in
        # multiple paths at the time of wheel repair. Therefore, it should not be
        # built with an RPATH since it is copied to the install prefix.
        if libdir == "SuiteSparse_config":
            # if in CI, set RPATH to the install directory for SuiteSparse_config
            # dylibs to find libomp.dylib when repairing the wheel
            if os.environ.get("CIBUILDWHEEL") == "1":
                env["CMAKE_OPTIONS"] = (
                    f"-DCMAKE_INSTALL_PREFIX={install_dir} -DCMAKE_INSTALL_RPATH={install_dir}/lib"
                )
            else:
                env["CMAKE_OPTIONS"] = f"-DCMAKE_INSTALL_PREFIX={install_dir}"
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
        if platform.processor() == "arm":
            OpenMP_C_FLAGS = (
                "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
            )
            OpenMP_C_LIB_NAMES = "omp"
            OpenMP_omp_LIBRARY = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
        elif platform.processor() == "i386":
            OpenMP_C_FLAGS = "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
            OpenMP_C_LIB_NAMES = "omp"
            OpenMP_omp_LIBRARY = "/usr/local/opt/libomp/lib/libomp.dylib"
        else:
            raise NotImplementedError(
                f"Unsupported processor architecture: {platform.processor()}. "
                "Only 'arm' and 'i386' architectures are supported."
            )

        # Don't pass the following args to CMake when building wheels. We set a custom
        # OpenMP installation for macOS wheels in the wheel build script.
        # This is because we can't use Homebrew's OpenMP dylib due to the wheel
        # repair process, where Homebrew binaries are not built for distribution and
        # break MACOSX_DEPLOYMENT_TARGET. We use a custom OpenMP binary as described
        # in CIBW_BEFORE_ALL in the wheel builder CI job.
        # Check for CI environment variable to determine if we are building a wheel
        if os.environ.get("CIBUILDWHEEL") != "1":
            print("Using Homebrew OpenMP for macOS build")
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
            f"Unsupported operating system: {platform.system()}. This script currently supports only Linux and macOS."
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
    description="Download, compile and install Sundials and SuiteSparse."
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
    sundials_found, suitesparse_found = False, False
else:
    # Check whether the libraries are installed
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
        install_suitesparse(download_dir)
        install_sundials(download_dir, install_dir)
    else:
        if not sundials_found:
            # Only SUNDIALS is missing, download and install it
            parallel_download([(SUNDIALS_URL, SUNDIALS_CHECKSUM)], download_dir)
            install_sundials(download_dir, install_dir)
        if not suitesparse_found:
            # Only SuiteSparse is missing, download and install it
            parallel_download([(SUITESPARSE_URL, SUITESPARSE_CHECKSUM)], download_dir)
            install_suitesparse(download_dir)
