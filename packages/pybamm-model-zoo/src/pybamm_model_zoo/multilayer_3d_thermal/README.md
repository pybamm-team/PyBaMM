# MultiLayer3DThermalSPM

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/multilayer_3d_thermal.json)

## Summary

A pouch cell stack resolved through its thickness. The stack's
`num_physical_layers` unit cells are lumped into `num_subdivisions` zones, and
each zone runs its own electrochemistry for its unit cells in parallel: an SPM
(`MultiLayer3DThermalSPM`), an SPMe (`MultiLayer3DThermalSPMe`), or a DFN
(`MultiLayer3DThermalDFN`). Every zone also carries its own temperature field
`T_i(x, y, z)` on a 3D finite-element mesh, whose volume average its kinetics and
transport see. Adjacent zones exchange heat through a thermal contact
resistance, and every exposed face is cooled convectively. The zones connect in
parallel, sharing the terminal voltage while their current fractions are solved
for, or in series.

Prefer it to `pybamm.lithium_ion.Basic3DThermalSPM` when the question is about
the stack's thickness: a cooling plate on one face, a degraded layer, anything
that makes the layers of a pouch run at different temperatures. That model has
one temperature field and one electrochemistry for the whole cell, so it cannot
move current between layers. Prefer PyBaMM's `"lumped"` or `"x-full"` thermal
options when the stack is close to isothermal through its thickness; they are
far cheaper.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo

model = zoo.load("MultiLayer3DThermalSPM")(num_physical_layers=24, num_subdivisions=6)
parameter_values = model.apply_stack_scaling(model.default_parameter_values)
parameter_values.update(
    {
        "Left face heat transfer coefficient [W.m-2.K-1]": 50.0,
        model.CONTACT_RESISTANCE_PARAM: 1e-2,
    }
)
simulation = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at 3C until 2.8 V"]),
)
solution = simulation.solve()
print(solution["Temperature spread [K]"](solution.t[-1]))
```

The SPMe and DFN variants take the same arguments:

```python
from pybamm_model_zoo.multilayer_3d_thermal import (
    MultiLayer3DThermalDFN,
    MultiLayer3DThermalSPMe,
)
```

`examples/run_multilayer_3d_thermal.py` cools a stack through one face and
follows the current as it moves to the warm side and back.
`examples/compare_electrochemistry.py` runs the same stack with all three
electrochemistries.

### Arguments

* `num_physical_layers` — unit cells in the stack, at least 2.
* `num_subdivisions` — zones, at least 2 and a divisor of `num_physical_layers`;
  defaults to one zone per unit cell. Fewer zones are cheaper, and under uniform
  cooling give the same answer.
* `connection` — `"parallel"` or `"series"`. The unit cells within a zone are
  always in parallel.
* `mesh_h` — target element size of each zone's mesh, as for
  `pybamm.ScikitFemGenerator3D`.

### Parameters

The parameter set describes one unit cell, and `"Current function [A]"` is the
current through the whole stack. `apply_stack_scaling` multiplies `"Nominal cell
capacity [A.h]"` by the unit cells that share that current — all of them in
parallel, one zone's in series — so that an experiment's C-rate is each unit
cell's own. Leave `"Number of electrodes connected in parallel to make a cell"`
at 1: the stack's unit cells are this model's layers, and setting it too would
divide the current twice.

`default_parameter_values` and `apply_stack_scaling` add these where the
parameter set lacks them:

* `"Inter-layer thermal contact resistance [K.m2.W-1]"`, default `1e-4`: close to
  perfect contact, but large enough to keep the coupling well posed.
* `"<Face> face heat transfer coefficient [W.m-2.K-1]"` for the `Left` (`x = 0`,
  zone 0's outer face), `Right`, `Front`, `Back`, `Bottom`, and `Top` faces,
  default `10`.

## Variables

Per zone `i`, each prefixed `"Layer i "`:

* `temperature [K]` — the 3D field — and `average temperature [K]`.
* `heat generation [W.m-3]`, per unit volume of a unit cell.
* `voltage [V]`, `current [A]` (the zone's), `per-unit-cell current [A]`, and, in
  parallel, `current fraction`.
* `X-averaged negative particle concentration [mol.m-3]` and the positive, and
  each electrode's `particle surface stoichiometry`.
* SPM and SPMe: `surface open-circuit voltage [V]`.
* SPMe and DFN: `electrolyte concentration [mol.m-3]`, `X-averaged electrolyte
  concentration [mol.m-3]`, and `total lithium in electrolyte per unit cell [mol]`.
* SPMe: `X-averaged concentration overpotential [V]`, `X-averaged electrolyte
  ohmic losses [V]`, and `X-averaged solid phase ohmic losses [V]`.
* DFN: the particle concentrations `c_s(r, x)`, `electrolyte potential [V]`, and
  each electrode's `electrode potential [V]`.

For the stack: `Voltage [V]`, `Current [A]`, `Stack-averaged temperature [K]`
(also as `Volume-averaged cell temperature [K]`), `Maximum layer-averaged
temperature [K]`, `Minimum layer-averaged temperature [K]`, and `Temperature
spread [K]`.

## Validation

All of these run in `tests/test_multilayer_3d_thermal.py`, with `Marquis2019`:

* Held isothermal by cooling every face at `1e4 W.m-2.K-1`, a symmetric two-zone
  stack matches `pybamm.lithium_ion.SPM`, `SPMe`, and `DFN` to within 1 mV over a
  30 minute discharge (measured 0.03, 0.34, and 0.15 mV).
* Insulated on every face, the stack stores the heat it generates to within 0.1%
  (measured 0.002%). This pins the interface coupling: heat leaves one zone
  exactly as it enters the next.
* Without entropic heat, each SPM and SPMe zone dissipates exactly its current
  times the voltage it loses below its surface open-circuit voltage.
* The SPMe and DFN electrolytes conserve lithium to `rtol=1e-9`.
* A symmetric two-zone parallel stack splits its current evenly, with no
  temperature spread, for all three electrochemistries.
* Cooled through one face, the zones warm monotonically away from it, and two
  thirds of the way through a 2C discharge the warmer zones carry monotonically
  more of the current.
* Six unit cells in two zones reproduce six in six under uniform cooling: the
  voltage to 10 µV and the stack temperature to 1 mK.
* After `apply_stack_scaling`, 1C drives every unit cell at the parameter set's
  own 1C current, in parallel and in series.

Not validated, and worth knowing before relying on it:

* Nothing has been compared with an experiment or a published multilayer model.
* The effective thermal conductivity is PyBaMM's `lambda_eff`, the
  thickness-weighted average of the layers including the current collectors,
  and it is applied in every direction. That is the in-plane conductivity;
  through the stack the layers conduct in series, so this overstates conduction
  across the zones, and the contact resistance is the only lever against it.
* The effective heat capacity and conductivity include the current collectors,
  but each unit cell spans only `L_x`, its electrodes and separator.
* A zone's electrochemistry sees only its own volume-averaged temperature, so the
  field's in-plane gradients carry heat but do not feed back into the kinetics.
* Each zone generates its heat uniformly over its volume. The SPM and SPMe
  dissipate their transport losses as current times voltage drop, and no model
  includes the heat of mixing.
* Tabs, current collector resistance, and in-plane current distribution are not
  modelled. Model options other than `"cell geometry"` are accepted but have no
  effect, as in PyBaMM's basic models: each zone's equations are written out.

## Citation

See `CITATION.bib`. Cite `MultiLayer3DThermal2026` for this entry, and
`Marquis2019` for the SPM and SPMe zones or `Doyle1993` for the DFN zones. Each
is registered when the model is instantiated, so `pybamm.print_citations()`
lists them.

## Maintainer

mleot (@mleot) — tier: community. First contributed in
[#5489](https://github.com/pybamm-team/PyBaMM/pull/5489).

Code owners need write access to the repository, so the PyBaMM maintainers
(@pybamm-team/maintainers) own this folder in `.github/CODEOWNERS` and approve
pull requests to it. Support for the model rests with its maintainer.
