#!/bin/bash
set -e -x

# GitHub runners add "-e LD_LIBRARY_PATH" option to "docker run",
# overriding default value of LD_LIBRARY_PATH in manylinux image. This
# causes libcrypt.so.2 to be missing (it lives in /usr/local/lib)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# CLI arguments
PY_VERSIONS=$1

git clone https://github.com/pybind/pybind11.git /github/workspace/pybind11
# Compile wheels
arrPY_VERSIONS=(${PY_VERSIONS// / })
for PY_VER in "${arrPY_VERSIONS[@]}"; do
    # Update pip
    /opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip

    # Build wheels
    /opt/python/"${PY_VER}"/bin/pip wheel /github/workspace/ -w /github/workspace/wheelhouse/ --no-deps || { echo "Building wheels failed."; exit 1; }
done
ls -l /github/workspace/wheelhouse/

# Bundle external shared libraries into the wheels
for whl in /github/workspace/wheelhouse/*-linux*.whl; do
    auditwheel repair "$whl" --plat "${PLAT}" -w /github/workspace/dist/ || { echo "Repairing wheels failed."; auditwheel show "$whl"; exit 1; }
done

echo "Succesfully built wheels:"
ls -l /github/workspace/dist/
