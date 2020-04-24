## Prerequisites

To use and/or contribute to PyBaMM, you must have Python 3.6 or 3.7 installed (note that 3.8 is not yet supported).

To install Python 3 on Debian-based distribution (Debian, Ubuntu, Linux mint), open a terminal and run
```bash
sudo apt update
sudo apt install python3
```
On Fedora or CentOS, you can use DNF or Yum. For example
```bash
sudo dnf install python3
```
On Mac OS distributions, you can use `homebrew`.
First [install `brew`](https://docs.python-guide.org/starting/install3/osx/):

```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

then follow instructions in link on adding brew to path, and run

```bash
brew install python3
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

PyBaMM's dependencies (such as `numpy`, `scipy`, etc) will be installed automatically when you install PyBaMM using `pip`.

For an introduction to virtual environments, see (https://realpython.com/python-virtual-environments-a-primer/).

#### Optional - scikits.odes solver
Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).

**A pre-requisite** is the installation of a BLAS library (such as [openblas](https://www.openblas.net/)).
On Ubuntu/debian
```
sudo apt install libopenblas-dev
```
After installing PyBaMM, the following command can be used to automatically install `scikits.odes`
and its dependencies
```
$ pybamm_install_odes --install-sundials
```
The  `--install-sundials` option is used to activate automatic downloads and installation of the sundials library, which is required by `scikits.odes`.

### Developer install

If you wish to contribute to PyBaMM, you should get the latest version from the GitHub repository.
To do so, you must have Git and graphviz installed. For instance run

```bash
sudo apt install git graphviz
```

on Debian-based distributions, or

```bash
brew install git graphviz
```

on Mac OS.

To install PyBaMM, the first step is to get the code by cloning this repository

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```
Then, to install PyBaMM as a [developer](CONTRIBUTING.md), type

```bash
pip install -e .[dev,docs]
```

**KLU sparse solver** If you wish so simulate large systems such as the 2+1D models, we recommend employing a sparse solver.
PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials, but it is
unlikely to be installed as you may not have all the dependencies available. If you wish to install the KLU from the PyBaMM sources, see [compiling the KLU sparse solver](compiling_KLU.md).

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
