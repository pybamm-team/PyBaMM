## Prerequisites

You'll need the following requirements:

- Python 3.6+
- Git (`git` package on Ubuntu distributions)
- Python libraries: `venv` (`python3-venv` package on Ubuntu distributions)

You can get these on a Debian based distribution using `apt-get`

```bash
sudo apt-get install python3 git-core python3-venv
```

## Install PyBaMM

The first step is to get the code by cloning this repository

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```

The safest way to install PyBaMM is to do so within a virtual environment ([introduction
to virtual environments](https://realpython.com/python-virtual-environments-a-primer/)).
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

PyBaMM has the following python libraries as dependencies: `numpy`, `scipy`, `pandas`,
`matplotlib`. These, along with PyBaMM, can easily be installed using `pip`. First, make
sure you have activated your virtual environment as above, and that you have the latest
version of pip installed:

```bash
pip install --upgrade pip
```

Then navigate to the path where you downloaded PyBaMM to (you will already be in the
correct location if you followed the instructions above), and install both PyBaMM and
its dependencies by typing:

```bash
pip install .
```

Or, if you want to install PyBaMM as a [developer](CONTRIBUTING.md), use

```bash
pip install -e .[dev,docs]
```

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

- Python header files (`python-dev/python3-dev` on Debian/Ubuntu-based distributions)
- C compiler
- Fortran compiler (e.g. gfortran)
- BLAS/LAPACK install (OpenBLAS is recommended by the scikits.odes developers)
- CMake (for building Sundials)
- Sundials 3.1.1

You can install these on Ubuntu or Debian using apt-get:

```bash
sudo apt-get install python3-dev gfortran gcc cmake libopenblas-dev
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

After this, you will need to set your `LD_LIBRARY_PATH` to point to the sundials
library:

```bash
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

You may wish to put this last line in your `.bashrc` or virtualenv `activate` script,
which will save you needing to set your `LD_LIBRARY_PATH` every time you log in. For
example, to add this line to your `.bashrc` you can type:

```bash
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions.

## Troubleshooting

**Problem:** I've made edits to source files in PyBaMM, but these are not being used 
when I run my Python script.

**Solution:** Make sure you have installed PyBaMM using the `-e` flag, i.e. `pip install 
-e .`. This sets the installed location of the source files to your current directory.

**Problem:** When running `python run-tests.py --quick`, gives error `FileNotFoundError: 
[Errno 2] No such file or directory: 'flake8': 'flake8`.

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

