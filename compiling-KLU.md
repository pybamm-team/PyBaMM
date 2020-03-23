# PyBaMM developper install - The KLU sparse solver
If you wish so simulate large systems such as the 2+1D models, we recommend employing a
sparse solver.
PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials.

When installing PyBaMM from source (e.g. from the GitHub repository), the KLU sparse solver must
be compiled.
_In the following we call "project directory" the directory containing the file `setup.py`._

Running `pip install .` in the project directory will result in a attempt to compile the
KLU solver.
Should this compilation ecounter errors (most likely because you do not have the required dependencies
installed), it will abort.

To install the KLU solver as part of PyBaMM, you will need:
+ A C++ compiler (e.g. `g++`)
+ The python 3 header files
+ [CMake](https://cmake.org/)
+ A BLAS implementation (e.g. [openblas](https://www.openblas.net/))
+ [pybind11](https://github.com/pybind/pybind11)
+ [sundials](https://computing.llnl.gov/projects/sundials)
+ [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html)

The first four dependencies should be available through your package manager.
Please raise an issue if not!
For instance on Ubuntu/Debian:
```bash
apt update
apt install python3-dev gcc cmake libopenblas-dev
```

## pybind11
The pybin11 source should be accessible upon compilation.
Simply clone the GitHub repository, for example:
```bash
mkdir third-party
cd third-party
git clone https://github.com/pybind/pybind11.git
```

## SuiteSparse and sundials 
### Method 1 - Using the convenience script
The PyBaMM repository contains a script `scripts/setup_KLU_module_build.py` that automatically
downloads, extracts, compiles and installs the two libraries.
The resulting files are located inside the PyBaMM project directory under `KLU_module_deps/`.
```
python scripts/setup_KLU_module_build.py
```

### Method 2 - Install from package manager
On most current linux distributions and macOS, a recent enough version of 
the suitesparse source package is available through the package manager.
For instance on Fedora
```
yum install libsuitesparse-dev
```
The PyBaMM KLU solver depends on the version of sundials being _at least_ 4.0.0, and unfortunately such requirement is _not available though most distribution's package managers_.
As a result the sundials library must be compiled manually.
Come back! It's not that difficult.

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
to the correct installation location on your system.
If you installed SuiteSparse through your package manager, this is likely to be something of 
the type:
```
-DKLU_INCLUDE_DIR=/usr/include/suitesparse\
-DKLU_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu\
```
By default sundials will be installed on your system under `/usr/local` (this varies depending on the 
distribution).
Should you wish to install sundials in a specific location, set the following variable
```
-DCMAKE_INSTALL_PREFIX=install/location\
```
Finally, you can build the library:
```
make install
```
You may be asked to run this command as a super-user, depending on the installation location.

### Alternative installation location
By default, commands relying on the `setup.py` like `pip install .` and  `python setup.py install`
will look for SuiteSparse and sundials in directories `KLU_module_deps/SuiteSparse-5.6.0` and 
`KLU_module_deps/sundials5`, respectively.
If not found, the system libraries (under `/usr/local/` or `/usr/`) are searched.

It is always possible to install the libraries to a different location, and specify this location
whe ninvoking the above commands.
For example

```
python setup.py install --suitesparse-root=path/to/suitesparse
```
or 
```
pip install . --install-option="--sundials-root=path/to/sundials"
```

## Check that the solver is correctly installed
If the KLU sparse solver is correcty installed, then the following command
should return `True`.
```
$ python -c "import pybamm; print(pybamm.have_idaklu())"
```
