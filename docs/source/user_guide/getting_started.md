# Getting Started

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:

```python
import pybamm

model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()
```

or simulate an experiment such as a constant-current discharge followed by a constant-current-constant-voltage charge:

```python
import pybamm

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        )
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
```

However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type, number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](https://github.com/pybamm-team/PyBaMM/blob/main/examples/scripts/DFN.py) or [notebook](https://github.com/pybamm-team/PyBaMM/blob/main/docs/source/examples/notebooks/models/DFN.ipynb)).

For new users we recommend the [Getting Started](https://github.com/pybamm-team/PyBaMM/tree/main/docs/source/examples/notebooks/getting_started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM, and can either be downloaded and used locally, or used online through [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/).

Further details can be found in a number of [detailed examples](https://github.com/pybamm-team/PyBaMM/blob/main/docs/source/examples/index.rst), hosted on
GitHub. In addition, full details of classes and methods can be found in the [](api_docs).
Additional supporting material can be found
[here](https://github.com/pybamm-team/pybamm-supporting-material/).

## Licence and commercial use

PyBaMM is released under the [BSD 3-Clause licence](https://github.com/pybamm-team/PyBaMM/blob/main/LICENSE.txt), a permissive open-source licence.
You may use, modify and redistribute PyBaMM, including in commercial and closed-source products, provided that:

- redistributions keep the copyright notice, the list of conditions and the disclaimer from `LICENSE.txt`, in the source code or in the documentation shipped with binaries;
- the name of PyBaMM or its contributors is not used to endorse or promote derived products without prior written permission.

PyBaMM is provided "as is", without warranty of any kind.
See [`LICENSE.txt`](https://github.com/pybamm-team/PyBaMM/blob/main/LICENSE.txt) for the full terms.
