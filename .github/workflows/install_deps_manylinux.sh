# Install CMake and wget
yum remove cmake -y
/opt/python/cp36-cp36m/bin/python -m pip install cmake wget
ln /opt/python/cp36-cp36m/bin/cmake /usr/bin/cmake

git clone https://github.com/pybind/pybind11.git /github/workspace/pybind11

# Download, build and install SuiteSparse + Sundials with KLU enabled
yum install openblas-devel -y
/opt/python/cp36-cp36m/bin/python /github/workspace/scripts/setup_KLU_module_build.py

