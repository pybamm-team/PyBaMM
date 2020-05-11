# PyBaMM developer install - The KLU sparse solver
If you wish to try a different DAE solver, PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials.
This solver comes as a C++ python extension module.
Therefore, when installing PyBaMM from source (e.g. from the GitHub repository), the KLU sparse solver module must be compiled.
Running `pip install .` or `python setup.py install ` in the PyBaMM directory will result in a attempt to compile the KLU module.

Note that if CMake of pybind11 are not found (see below), the installation of PyBaMM will carry on, however skipping
the compilation of the `idaklu` module. This allows developers that are not interested in the KLU module to install PyBaMM from source without having to install the required dependencies.

To build the KLU solver, the following dependencies are required:

- A C++ compiler (e.g. `g++`)
- A Fortran compiler (e.g. `gfortran`)
- The python 3 header files
- [CMake](https://cmake.org/)
- A BLAS implementation (e.g. [openblas](https://www.openblas.net/))
- [pybind11](https://github.com/pybind/pybind11)
- [sundials](https://computing.llnl.gov/projects/sundials)
- [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html)

The first four should be available through your favourite package manager.
On Debian-based GNU/Linux distributions:
```bash
apt update
apt install python3-dev gcc gfortran cmake libopenblas-dev
```

## pybind11
The pybind11 source directory should be located in the PyBaMM project directory at the time of
compilation.
Simply clone the GitHub repository, for example:
```bash
# In the PyBaMM project dir (next to setup.py)
git clone https://github.com/pybind/pybind11.git
```
## SuiteSparse and sundials
### Method 1 - Using the convenience script
The PyBaMM repository contains a script `scripts/setup_KLU_module_build.py` that automatically
downloads, extracts, compiles and installs the two libraries.

First install the Python `wget` module
```
pip install wget
```
Then execute the script
```
# In the PyBaMM project dir (next to setup.py)
python scripts/setup_KLU_module_build.py
```
The above will install the required component of SuiteSparse and Sundials in your home directory under
`~/.local/`.
Note that you can provide the option  `--install-dir=<install/path>` to install both libraries to
an alternative location. If `<install/path>` is not absolute, it will be interpreted as relative to the PyBaMM project directory.

Finally, reactivate your virtual environment by running
```
source $(VIRTUAL_ENV)/bin/activate
```
Alternatively, you update the `LD_LIBRARY_PATH` environment variable as follows
```
export LD_LIBRARY_PATH=$(HOME)/.local:$LD_LIBRARY_PATH
```
The above export statement will be ran automatically the next time you activate you python virtual environment.

If did not run the convenience script inside a python virtual environment, execute you bash config file
```
source ~/.bashrc
```
(or start a new shell).

Build files are located inside the PyBaMM project directory under `KLU_module_deps/`.
Feel free to remove this directory once everything is installed correctly.

### Method 2 - Compiling Sundials (advanced)

#### SuiteSparse
On most current linux distributions and macOS, a recent enough version of
the suitesparse source package is available through the package manager.
For instance on Fedora
```
yum install libsuitesparse-dev
```

#### Sundials
The PyBaMM KLU solver requires Sundials >= 4.0. Because most Linux distribution provide older versions through
their respective package manager, it is recommended to build and install Sundials manually.

First, download and extract the sundials 5.0.0 source
```
wget https://computing.llnl.gov/projects/sundials/download/sundials-5.0.0.tar.gz .
tar -xvf sundials-5.0.0.tar.gz
```
Then, create a temporary build directory and navigate into it
```
mkdir build_sundials
cd build_sundials
```
You can now configure the build, by running
```
cmake -DLAPACK_ENABLE=ON\
      -DSUNDIALS_INDEX_SIZE=32\
      -DBUILD_ARKODE=OFF\
      -DBUILD_CVODE=OFF\
      -DBUILD_CVODES=OFF\
      -DBUILD_IDAS=OFF\
      -DBUILD_KINSOL=OFF\
      -DEXAMPLES_ENABLE:BOOL=OFF\
      -DKLU_ENABLE=ON\
      -DKLU_INCLUDE_DIR=path/to/suitesparse/headers\
      -DKLU_LIBRARY_DIR=path/to/suitesparse/libraries\
      ../sundials-5.0.0
```
Be careful set the two variables `KLU_INCLUDE_DIR` and `KLU_LIBRARY_DIR`
to the correct installation location of the SuiteSparse libary on your system.
If you installed SuiteSparse through your package manager, this is likely to be something similar to:
```
-DKLU_INCLUDE_DIR=/usr/include/suitesparse\
-DKLU_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu\
```
By default, Sundials will be installed on your system under `/usr/local` (this varies depending on the
distribution).
Should you wish to install sundials in a specific location, set the following variable
```
-DCMAKE_INSTALL_PREFIX=install/location\
```
Finally, build the library:
```
make install
```
You may be asked to run this command as a super-user, depending on the installation location.

#### Alternative installation location
By default, it is assumed that the SuiteSparse and Sundials libraries are installed in your home directory
under `~/.local`.
If you installed the libraries to (a) different location(s), you must set the options
`suitesparse-root` or/and `sundials-root` when installing PyBaMM.
Examples:

```
python setup.py install --suitesparse-root=path/to/suitesparse
```
or
```
pip install . --install-option="--sundials-root=path/to/sundials"
```

## (re)Install PyBaMM to build the KLU solver
If the above dependencies are correctly installed, any of the following commands
will install PyBaMM with the `idaklu` solver module:
```
pip install .
pip install -e .
python setup.py install
python setup.py develop
...
```
Note that it doesn't matter if pybamm is already installed. The above commands will update your exisitng installation by adding the `idaklu` module.

## Check that the solver is correctly installed
If you install PyBaMM in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) using the `-e` pip switch or if you use the `python setup.py install` command, a log file will be located in the project directory (next to the `setup.py` file).
```
cat setup.log
020-03-24 11:33:50,645 - PyBaMM setup - INFO - Starting PyBaMM setup
2020-03-24 11:33:50,653 - PyBaMM setup - INFO - Not running on windows
2020-03-24 11:33:50,654 - PyBaMM setup - INFO - Could not find CMake. Skipping compilation of KLU module.
2020-03-24 11:33:50,655 - PyBaMM setup - INFO - Could not find pybind11 directory (/io/pybind11). Skipping compilation of KLU module.
```

If the KLU sparse solver is correctly installed, then the following command
should return `True`.
```
$ python -c "import pybamm; print(pybamm.have_idaklu())"
```
