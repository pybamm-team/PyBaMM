# PyBaMM

[![travis](https://travis-ci.org/pybamm-team/PyBaMM.svg?branch=master)](https://travis-ci.org/pybamm-team/PyBaMM)
[![Build status](https://ci.appveyor.com/api/projects/status/xdje8jnhuj0ye1jc/branch/master?svg=true)](https://ci.appveyor.com/project/martinjrobins/pybamm/branch/master)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master?filepath=examples%2Fnotebooks)
[![black_code_style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

PyBaMM (Python Battery Mathematical Modelling) solves physics-based electrochemical DAE models by using state-of-the-art automatic differentiation and numerical solvers. The Doyle-Fuller-Newman model can be solved in under 0.1 seconds, while the reduced-order Single Particle Model and Single Particle Model with electrolyte can be solved in just a few milliseconds. Additional physics can easily be included such as thermal effects, fast particle diffusion, 3D effects, and more. All models are implemented in a flexible manner, and a wide range of models and parameter sets (NCA, NMC, LiCoO2, ...) are available. There is also functionality to simulate any set of experimental instructions, such as CCCV or GITT, or specify drive cycles.

## How do I use PyBaMM?

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:
```python3
import pybamm
model = pybamm.lithium_ion.DFN() # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve()
sim.plot()
```
or simulate an experiment such as CCCV:
```python3
import pybamm
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
```
However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type,  number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](examples/scripts/DFN.py) or [notebook](examples/notebooks/models/dfn.ipynb)).

For new users we recommend the [Getting Started](examples/notebooks/Getting%20Started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM.

Further details can be found in a number of [detailed examples](examples/notebooks/README.md), hosted here on
github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/),
hosted on [Read The Docs](readthedocs.io). A set of slides giving an overview of PyBaMM
can be found
[here](https://github.com/pybamm-team/pybamm_summary_slides/blob/master/pybamm.pdf).

For further examples, see the list of repositories that use PyBaMM [here](https://github.com/pybamm-team/pybamm-example-results)

## How can I obtain & install PyBaMM?

### Linux

For instructions on installing PyBaMM on Debian-based distributions, please see [here](INSTALL-LINUX-MAC.md).

### Mac OS

For instructions on installing PyBaMM on Mac OS distributions, please see [here](INSTALL-LINUX-MAC.md).

### Windows

For instructions on installing PyBaMM on Windows distributions, please see [here](INSTALL-WINDOWS.md). If you want to install the optional solvers (such as scikits-odes and KLU solvers), install PyBaMM on the Windows Subsystem for Linux following the instructions [here](INSTALL-WINDOWS-WSL.md) 

## Citing PyBaMM

If you use PyBaMM in your work, please cite our paper

> Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2020). Python Battery Mathematical Modelling (PyBaMM). _ECSarXiv. February, 7_.

You can use the bibtex

```
@article{sulzer2020python,
  title={Python Battery Mathematical Modelling (PyBaMM)},
  author={Sulzer, Valentin and Marquis, Scott G and Timms, Robert and Robinson, Martin and Chapman, S Jon},
  journal={ECSarXiv. February},
  volume={7},
  year={2020}
}
```

We would be grateful if you could also cite the relevant papers. These will change depending on what models and solvers you use. To find out which papers you should cite, add the line

```python3
pybamm.print_citations()
```

to the end of your script. This will print bibtex information to the terminal; passing a filename to `print_citations` will print the bibtex information to the specified file instead. A list of all citations can also be found in the [citations file](pybamm/CITATIONS.txt). In particular, PyBaMM relies heavily on [CasADi](https://web.casadi.org/publications/).

## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
