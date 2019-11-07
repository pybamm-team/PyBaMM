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

CasADi's DAE solvers are included by default in PyBaMM, but two additional DAE solvers (`scikits.odes` and `KLU`) can be optionally installed as well. 
In particular, the [KLU sparse solver](INSTALL-KLU.md) is recommended for solving the DFN.

### Sundials with KLU sparse solver

If you wish so simulate large systems such as the 2+1D models, we recommend employing a
sparse solver. PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials.
[Installation instructions](INSTALL-KLU.md).

### [scikits.odes](https://github.com/bmcage/odes)

A python wrapper for the SUNDIALS ODE and DAE integrators. [Installation instructions](INSTALL-SCIKITS.md).

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
