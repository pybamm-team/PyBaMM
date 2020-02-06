## Prerequisites

To use and/or contribute to PyBaMM, you must have Python 3.6 or above installed.
To install Python 3 on Debian-based distribution (Debian, Ubuntu, Linux mint), open a terminal and run
```bash
sudo apt update
sudo apt install python3
```
On Fedora or CentOS, you can use DNF or Yum. For example
```bash
sudo dnf install python3
```

## Install PyBaMM

### User install
We recommend to install PyBaMM within a virtual environment, in order not
to alter any distribution python files.
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

PyBaMM can be installed via pip:
```bash
pip install pybamm
```


PyBaMM has the following python libraries as dependencies: `numpy`, `scipy`, `pandas`,
`matplotlib`. These will be installed automatically when you install PyBaMM using `pip`,
following the instructions below. First, make sure you have activated your virtual
environment as above, and that you have the latest version of pip installed:

Then navigate to the path where you downloaded PyBaMM to (you will already be in the
correct location if you followed the instructions above), and install both PyBaMM and
its dependencies by typing:

```bash
pip install pybamm
```
For an introduction to virtual environments, see (https://realpython.com/python-virtual-environments-a-primer/).

### developer install

If you wish to contribute to PyBaMM, you should get the latest version from the GitHub repository.
To do so, you must have Git installed.
For instance run
```bash
sudo apt install git
```
on Debian-based distributions.

To install PyBaMM, the first step is to get the code by cloning this repository

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```
Then, install PyBaMM as a develop per with [developer](CONTRIBUTING.md), use

```bash
pip install -e .[dev,docs]
```

To check whether PyBaMM has installed properly, you can run the tests:

```bash
python3 run-tests.py --unit
```

Before you start contributing to PyBaMM, please read the [contributing guidelines](CONTRIBUTING.md).

## Uninstall PyBaMM
PyBaMM can be uninstalled by running
```bash
pip uninstall pybamm
```
in your virtual environment.

## Optional dependencies
The following instructions assume that you downloaded the PyBaMM source code and that all
commands are run from the PyBaMM root directory (`PyBaMM/`).
This can be done using `git`, running

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```
Alternatively, you can dowload the source code archive from [the PyBaMM GitHub repo](https://github.com/pybamm-team/PyBaMM.git) and extract it the location of your choice.

Ideally you should have the python package `wget` installed.
This allows for the automatic download of some of the dependencies has part of the installation process.
You can install it using (within your virtual environment)
```bash
pip install wget
```

### [scikits.odes](https://github.com/bmcage/odes)
Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).
The Sundials DAE solver is required to solve the DFN battery model in PyBaMM.

Before installing scikits.odes, you need to have installed:

- Python 3 header files (`python3-dev` on Debian/Ubuntu-based distributions)
- C compiler (e.g. `gcc`)
- Fortran compiler (e.g. `gfortran`)
- BLAS/LAPACK install (OpenBLAS is recommended by the scikits.odes developers)
- CMake (for building Sundials)

You can install these on Ubuntu or Debian using APT:

```bash
sudo apt update
sudo apt install python3-dev gfortran gcc cmake libopenblas-dev
```
To install scikits.odes, simply run
```bash
python setup.py install_odes
```
This commands will first download and build the SUNDIALS library, required to install and use `scikits.odes`.
This will download approximately 16MB of data and should only take a few minutes to compile.
Alternatively, you can specify a directory containing the source code of the Sundials library
```bash
python setup.py install_odes --sundials-src=<path/to/sundials/source>
```
By default, the sundials are installed in a `sundials` directory located at the root of the PyBaMM package.
You can provide another location by using the `--sundials-inst=<path/to/other/location>` option.

If you are installing `scikits.odes` within a virtual environment, the `activate` script will be automatically
updated to add the sundials installation directory to your `LD_LIBRARY_PATH`.
This is required in order to use `scikits.odes`.
As a consequence, after installation you should restart your virtual environment.

If you wish to install the scikits.odes outside of a virtual environment, your `.bashrc` will be modified instead.
After installation you should therefore run
```bash
source ~/.bashrc
```
Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions.

Finally, you can check your install by running
```bash
python -c "import pybamm; print(pybamm.have_scikits_odes())
```
### Sundials with KLU sparse solver
If you wish so simulate large systems such as the 2+1D models, we recommend employing a
sparse solver.
PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials.

#### Prerequisites
The requirements are the same than for the installation of `scikits.odes` (see previous section).
Additionally, the [pybind11 GitHub repository](https://github.com/pybind/pybind11.git) should be located in `PyBaMM/third-party/`.
First create a directory `third-party` and clone the repository:
```bash
mkdir third-party
cd third-party
git clone https://github.com/pybind/pybind11.git
cd ..
```
If you don't have `git` installed, you can download the code source manually from (https://github.com/pybind/pybind11).

#### Install the KLU solver
The KLU solver is can be installed _via_ the following command:
```bash
python setup.py install_klu
```
The previous command will download and install both the [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) and [SUNDIALS](https://computing.llnl.gov/projects/sundials) libraries.
This will download approximately 70MB of data and the compilation should only take a couple of minutes.
If the source for a library is already present on your system, you can specify its location using options `--suitesparse-src` or `--sundials-src`.
Example:
```bash
python setup.py install_klu --suitesparse-src=<path/to/suitesparse/source>
```
This will not download the SuiteSparse library and compile the source code located in `path/to/suitesparse/source`.
The sundials library will be downloaded.

Finally, you can check your install by running
```bash
python -c "import pybamm; print(pybamm.have_idaklu())
```

### Install everything
It is possible to install both `scikits.odes` and KLU solver using the command
```bash
python setup.py install_all
```
Note that options `--sundials-src`, `--sundials-inst` and  `suitesparse-src` are still usable
here.

You can make sure the install was successful by runing
Finally, you can check your install by running
```bash
python -c "import pybamm; print(pybamm.have_scikits_odes())
```
and

```bash
python -c "import pybamm; print(pybamm.have_idaklu())
```

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
check the instructions given above and make sure each command was successful.

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
