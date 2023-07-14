import os
import sys
import subprocess

# Folder containing hte file
cmake_list_dir = os.path.abspath(os.path.dirname(__file__))

build_dir = 'build'

cmake_args = [
    "-DCMAKE_BUILD_TYPE=DEBUG",
    "-DPYTHON_EXECUTABLE={}".format(sys.executable),
    "-DUSE_PYTHON_CASADI=TRUE",
]
if True:
    cmake_args.append(
        "-DSuiteSparse_ROOT={}".format(os.path.abspath("/home/jsb/.local"))
    )
if True:
    cmake_args.append(
        "-DSUNDIALS_ROOT={}".format(os.path.abspath("/home/jsb/.local"))
    )

build_env = os.environ
# build_env["vcpkg_root_dir"] = vcpkg_root_dir
# build_env["vcpkg_default_triplet"] = vcpkg_default_triplet
# build_env["vcpkg_feature_flags"] = vcpkg_feature_flags

subprocess.run(
    ["cmake", cmake_list_dir] + cmake_args, cwd=build_dir, env=build_env
)
