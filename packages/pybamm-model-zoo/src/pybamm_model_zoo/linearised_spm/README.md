# LinearisedSPM

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/linearised_spm.json)

## Summary

The Single Particle Model with both porous electrodes' open-circuit potentials
replaced by their tangent at the stoichiometry the simulation starts from,
`U(x) -> U(x_0) + U'(x_0) (x - x_0)`, and `"intercalation kinetics"` defaulting
to `"linear"`. The gradient is PyBaMM's symbolic derivative of whichever `U` the
parameter set supplies, so the model adds no parameters of its own.

This is the model for **GITT analysis**. Over a pulse short enough that the
particles stay near `x_0`, the voltage transient is the Weppner-Huggins form: it
varies as `sqrt(t)` with a slope set by the diffusivity and by `dU/dx`. Fitting
a diffusivity to a measured pulse needs `dU/dx`, which is normally read off a
separately measured titration curve; here the model holds it exactly and reports
it as `"<Domain> electrode open-circuit potential gradient [V]"`.

Prefer `pybamm.lithium_ion.SPM` for anything that ranges far from `x_0`. The
tangent is a local approximation, and over a full discharge it departs from the
real open-circuit voltage — which is exactly why this is a zoo entry and not a
core PyBaMM option ([#3187](https://github.com/pybamm-team/PyBaMM/issues/3187)).

This is also the model zoo's **reference entry**: it is deliberately the
smallest thing that is still a real model. Copy this folder as the starting
point for your own.

## Usage

```python
import numpy as np
import pybamm
import pybamm_model_zoo as zoo

model = zoo.load("LinearisedSPM")({"working electrode": "positive"})
parameter_values = model.default_parameter_values
parameter_values["Positive particle diffusivity [m2.s-1]"] = 1e-14

simulation = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    var_pts={**model.default_var_pts, "r_p": 200},
)
solution = simulation.solve([0, 5])

times = np.linspace(0.25, 5, 60)
slope = np.polyfit(np.sqrt(times), solution["Voltage [V]"](times), 1)[0]
gradient = solution["Positive electrode open-circuit potential gradient [V]"](0.25)
print(slope, gradient)
```

`examples/run_linearised_spm.py` completes the inversion and prints the fitted
diffusivity for three pulse lengths.

## Variables

Beyond SPM's own, per porous electrode:

* `"<Domain> electrode linearisation stoichiometry"` — the `x_0` linearised about.
* `"<Domain> electrode linearisation open-circuit potential [V]"` — `U(x_0)`.
* `"<Domain> electrode open-circuit potential gradient [V]"` — `dU/dx` at `x_0`,
  the `dE/dδ` a Weppner-Huggins fit needs.

## Validation

* Each electrode's reported open-circuit potential equals
  `U(x_0) + U'(x_0) (x - x_0)` to `rtol=1e-12`, from the model's own reported
  `x_0`, `U(x_0)`, and gradient.
* The reported gradient matches a central difference of the parameter set's OCP
  function to `rtol=1e-5`.
* At `t = 0` the bulk open-circuit voltage equals `pybamm.lithium_ion.SPM`'s to
  `rtol=1e-12`: linearising changes nothing at the point linearised about.
* A 5 s GITT pulse on a positive half cell recovers an imposed
  `1e-14 m2.s-1` diffusivity to within 5% through the Weppner-Huggins relation,
  and the residual error shrinks monotonically as the pulse shortens (−28.5% at
  80 s, −11.4% at 20 s, −2.9% at 5 s). That residual is the `sqrt(t)`
  approximation's own spherical-geometry bias, not an error in the model, which
  is why GITT pulses are kept short in practice.
* All of these run in `tests/test_linearised_spm.py`.

Not validated, and rejected rather than silently ignored: the
`"open-circuit potential"` option, since this model supplies its own — any value
other than `"single"` raises `pybamm.OptionError`. Nothing linearises the
kinetics' exchange current density or a planar electrode's plating potential, so
the model is not exactly linear in the state; over a short pulse both terms are
near constant.

## Citation

See `CITATION.bib`. Cite `WeppnerHuggins1977` for the method,
`PyBaMMModelZoo2026` for this entry, and `Marquis2019` for the underlying SPM;
all three are registered automatically, so `pybamm.print_citations()` lists them
after you use the model.

## Maintainer

The PyBaMM Team (@pybamm-team/maintainers) — tier: core
