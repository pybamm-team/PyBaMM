# Install IDA-KLU Solver

---
**Note**

This file provides installation instructions for either Ubuntu-based distributions or Mac OS distributions. Please read carefully which lines to run in each case.

---

If you are on a linux based distribution, a bash script has been provided which should
install everything for you correctly. Please note you will require the python header files, openblas,
a c compiler (e.g. gcc), cmake, and suitesparse all of which you should be able to install, either on ubuntu using

```bash
apt install python3-dev libopenblas-dev cmake gcc libsuitesparse-dev
```

or on a Mac OS distribution using brew (`python3-dev` is installed by `python3`):

```bash
brew install wget gcc cmake openblas suitesparse
```

You will likely need to prepend `sudo` to the above command.

## Installing KLU

### Option 1: install with script

We recommend that you first try to install KLU by running the script. From within the main PyBaMM directory type

```bash
source scripts/install_sundials_4.1.0.sh
```

Note that this script has only been tested on Ubuntu 18.04.3 LTS. If this works, skip to [the final section](#setting-library-path). Otherwise, try Option 2 below. If this script does not work for you, you can try following the step-by-step instructions below in Option 2.

## Option 2: install manually

#### Download and build Sundials 4.1.0

The KLU solver is interfaced using an updated version of Sundials so even if you have installed Sundials for use with Scikits.odes, you still need to install sundials here. If you want more information on the sundials installation please refer to the the ida_guide.pdf available at on the [sundials site](https://computing.llnl.gov/projects/sundials/sundials-software)

First, download Sundials 4.1.0 using

```bash
wget https://computing.llnl.gov/projects/sundials/download/sundials-4.1.0.tar.gz -O sundials-4.1.0.tar.gz
tar -xvf sundials-4.1.0.tar.gz
rm sundials-4.1.0.tar.gz
```

The cmake instructions provided with Sundials have trouble linking the required libraries related to the KLU solver, therefore we have provided a modified `CMakeLists.txt` file which fixes this. Copy this across into the sundials-4.1.0 folder, overwriting the old file, using

```bash
cp scripts/replace-cmake/CMakeLists.txt sundials-4.1.0/CMakeLists.txt
```

Now create a directory to build sundials in and set the install directory for sundials:

```bash
mkdir build-sundials-4.1.0
INSTALL_DIR=`pwd`/sundials4
```

Now enter the build directory, use cmake to generate the appropriate make files, and then build sundials and install sundials into the install directory using make:

```bash
cd build-sundials-4.1.0
cmake -DBLAS_ENABLE=ON\
      -DLAPACK_ENABLE=ON\
      -DSUNDIALS_INDEX_SIZE=32\
      -DBUILD_ARKODE=OFF\
      -DBUILD_CVODE=OFF\
      -DBUILD_CVODES=OFF\
      -DBUILD_IDAS=OFF\
      -DBUILD_KINSOL=OFF\
      -DEXAMPLES_ENABLE:BOOL=OFF\
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR\
      -DKLU_ENABLE=ON\
      ../sundials-4.1.0
make install
```

Now return to your PyBaMM home directory and remove the build-sundials-4.1.0 folder and the download folder:

```bash
cd ..
rm -rf build-sundials-4.1.0
rm -rf sundials-4.1.0
```

#### Install pybind11
To interface with Sundials which is written in C, we require pybind11. Clone the pybind11 repository whilst within a folder the third-party folder:

```bash
mkdir third-party
cd third-party
git clone https://github.com/pybind/pybind11.git
cd ..
```

You will also require pybind11 to be pip installed so from within your virtual enviroment (if you are using one) type:

```bash
pip install pybind11
```

#### Build the KLU wrapper
We now have all the tools to build a shared library to interface to the KLU solver. Within your PyBaMM home directory build the required Makefile using

```bash
cmake .
```

This will automatically find the headers for the latest version of python installed on your machine. If you are using an older version (e.g python3.6) within your virtual environment, then you instead can use `cmake -DPYBIND11_PYTHON_VERSION=3.6 .`.

You can now simply run make to build the library (you can just run this command if you make some changes to klu.cpp)

```bash
make
```

To clean up you directory you can now remove the automatically generated cmake files:
```
rm -rf CMakeFiles
rm CMakeCache.txt
rm cmake_install.cmake
```

## Setting library path

After this, you will need to set your `LD_LIBRARY_PATH` (for Linux) or `DYLD_LIBRARY_PATH` (for Mac) to point to the sundials
library - for Linux:

```bash
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

or for Mac:

```bash
export DYLD_LIBRARY_PATH=$INSTALL_DIR/lib:$DYLD_LIBRARY_PATH
```

You may wish to put one of these lines in your `.bashrc` or virtualenv `activate` script,
which will save you needing to set your `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` every time you log in. For
example, to add this line to your virtual environment `env` you can type:

```bash
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> env/bin/activate
```

for Linux or 

```bash
echo "export DYLD_LIBRARY_PATH=$INSTALL_DIR/lib:\$DYLD_LIBRARY_PATH" >> env/bin/activate
```

for Mac.

You can also try installing the [scikits.odes solver](INSTALL_SCIKITS.md) if you haven't already done so, but you only need one DAE solver.