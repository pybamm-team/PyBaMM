## Prerequisites

You'll need the following requirements:

- Python 3.6+
- Git (`git` package on Ubuntu distributions)
- Python libraries: `venv` (`python3-venv` package on Ubuntu distributions)
- Python graphical user interface (python3-tk)
- Graph visualization software (graphviz)

You can get these on a Debian based distribution using `apt-get`

```bash
sudo apt-get install python3 git-core python3-venv python3-tk graphviz
```

or on Mac OS based distributions by [installing `brew`](https://docs.python-guide.org/starting/install3/osx/):

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

following instructions in link on adding brew to path, and then running

```bash
brew install python3 git graphviz
```

## Install PyBaMM

The first step is to get the code by cloning this repository

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```

The safest way to install PyBaMM is to do so within a virtual environment ([a good introduction
to virtual environments to understand how these work](https://realpython.com/python-virtual-environments-a-primer/)).
To create a virtual environment `env` within your current directory type:

```bash
python3 -m venv env
```

You can then "activate" the environment using:

```bash
source env/bin/activate
```

Now all the calls to pip described below will install PyBaMM and its dependencies into
the environment `env`. When you are ready to exit the environment and go back to your
original system, just type:

```bash
deactivate
```

Whenever you close the terminal, or shut down, the environment is deactivated automatically. To go back into it, just run `source env/bin/activate` - this re-opens the same environment and the installs do not need to be rerun.

PyBaMM has some libraries, such as `numpy` and `scipy`, as dependencies (a full list is available [here](./setup.py), [here](./.requirements-docs.txt) or [here](https://github.com/pybamm-team/PyBaMM/network/dependencies)). These will be installed automatically when you install PyBaMM using `pip`,
following the instructions below. First, make sure you have activated your virtual 
environment as above, and that you have the latest version of pip installed:

```bash
pip install --upgrade pip
```

Then navigate to the path where you downloaded PyBaMM to (you will already be in the
correct location if you followed the instructions above), and install both PyBaMM and
its dependencies, either by typing:

```bash
pip install .
```

or if you want to install PyBaMM as a [developer](CONTRIBUTING.md) by typing

```bash
pip install -e .[dev,docs]
```

Note that it's fine to first run `pip install .` and then `pip install -e .[dev,docs]`; any packages that are already installed are skipped over.

To check whether PyBaMM has installed properly, you can run the tests:

```bash
python3 run-tests.py --unit
```

To uninstall PyBaMM, type

```bash
pip uninstall pybamm
```

## Optional dependencies

### [scikits.odes](https://github.com/bmcage/odes)

Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).
The Sundials DAE solver is required to solve the DFN battery model in PyBaMM.

Before installing scikits.odes, you need to have installed:

- Python header files (`python-dev/python3-dev` on Debian/Ubuntu-based distributions, comes with python3 by default in brew)
- C compiler
- Fortran compiler (e.g. gfortran, comes with gcc in brew)
- BLAS/LAPACK install (OpenBLAS is recommended by the scikits.odes developers)
- CMake (for building Sundials)
- Sundials 3.1.1 (see instructions below)

You can install these on Ubuntu or Debian using apt-get:

```bash
sudo apt-get install python3-dev gfortran gcc cmake libopenblas-dev
```

or on a Mac OS distribution using brew:

```bash
brew install gcc cmake openblas
```

To install Sundials 3.1.1, on the command-line type:

```bash
INSTALL_DIR=`pwd`/sundials
wget https://computation.llnl.gov/projects/sundials/download/sundials-3.1.1.tar.gz
tar -xvf sundials-3.1.1.tar.gz
mkdir build-sundials-3.1.1
cd build-sundials-3.1.1/
cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-3.1.1/
make install
rm -r ../sundials-3.1.1
```

Then install [scikits.odes](https://github.com/bmcage/odes), letting it know the sundials install location:

```bash
SUNDIALS_INST=$INSTALL_DIR pip install scikits.odes
```

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
example, to add this line to your `.bashrc` you can type:

```bash
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions.

### Sundials with KLU sparse solver

If you wish so simulate large systems such as the 2+1D models, we recommend employing a
sparse solver. PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials.
If you are on a linux based distribution, a bash script has been provided which should
install everything for you correctly. Please note you will require the python header files, openblas,
a c compiler (e.g. gcc), cmake, and suitesparse all of which you should be able to install, either on ubuntu using

```bash
apt install python3-dev libopenblas-dev cmake gcc libsuitesparse-dev
```

or on a Mac OS distribution using brew (`python3-dev` is installed by `python3`):

```bash
brew install gcc cmake openblas suitesparse
```

You will likely need to prepend `sudo` to the above command.

To install sundials with KLU, from within the main PyBaMM directory type

```bash
./scripts/install_sundials_4.1.0.sh
```

Note that this script has only been tested on Ubuntu 18.04.3 LTS. If this script does not work for you, you can try following the step-by-step instructions below:

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

## Troubleshooting

**Problem:** I've made edits to source files in PyBaMM, but these are not being used
when I run my Python script.

**Solution:** Make sure you have installed PyBaMM using the `-e` flag, i.e. `pip install
-e .`. This sets the installed location of the source files to your current directory.

**Problem:** When running `python run-tests.py --quick`, gives error `FileNotFoundError: 
[Errno 2] No such file or directory: `flake8`: `flake8`.

**Solution:** make sure you have included the `[dev,docs]` flags when you pip installed
PyBaMM, i.e. `pip install -e .[dev,docs]`

**Problem:** Errors when solving model `ValueError: Integrator name ida does not
exsist`, or `ValueError: Integrator name cvode does not exsist`.

**Solution:** This could mean that you have not installed `scikits.odes` correctly,
check the instrutions given above and make sure each command was successful.

One possibility is that you have not set your `LD_LIBRARY_PATH` to point to the sundials
library, type `echo $LD_LIBRARY_PATH` and make sure one of the directories printed out
corresponds to where the sundials libraries are located.

Another common reason is that you forget to install a BLAS library such as OpenBLAS
before installing sundials. Check the cmake output when you configured Sundials, it
might say:

```
-- A library with BLAS API not found. Please specify library location.
-- LAPACK requires BLAS
```

If this is the case, on a Debian or Ubuntu system you can install OpenBLAS using `sudo
apt-get install libopenblas-dev` and then re-install sundials using the instructions
above.
