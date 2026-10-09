# MultiLayer3DThermalSPM

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/multilayer_3d_thermal.json)

## Summary

A pouch cell stack resolved through its thickness. The stack's
`num_physical_layers` unit cells are lumped into `num_subdivisions` zones, and
each zone is PyBaMM's own model for its unit cells in parallel:
`pybamm.lithium_ion.SPM` (`MultiLayer3DThermalSPM`), `SPMe`
(`MultiLayer3DThermalSPMe`), or `DFN` (`MultiLayer3DThermalDFN`), built with the
options the stack is given. Particle phases, open-circuit potential models
(hysteresis included), intercalation kinetics, and every other electrochemical
option therefore behave exactly as they do in that model. Every zone also
carries its own temperature field `T_i(x, y, z)` on a 3D finite-element mesh,
which replaces the zone model's lumped temperature: the field's source is the
zone's total heating, and its volume average is the temperature the zone's
kinetics and transport see. Adjacent zones exchange heat by conduction through
the stack plus a contact resistance, and every exposed face is cooled
convectively. The zones connect in parallel, each with its current solved for,
or in series.

Prefer it to `pybamm.lithium_ion.Basic3DThermalSPM` when the question is about
the stack's thickness: a cooling plate on one face, a degraded layer, anything
that makes the layers of a pouch run at different temperatures. That model has
one temperature field and one electrochemistry for the whole cell, so it cannot
move current between layers. Prefer PyBaMM's `"lumped"` or `"x-full"` thermal
options when the stack is close to isothermal through its thickness; they are
far cheaper.

## Usage

Each zone's 3D mesh needs PyBaMM's finite-element extra, which the model's extra
installs:

```bash
pip install "pybamm-model-zoo[zoo-multilayer-3d-thermal]"
```

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
  always in parallel. In series, the stack stops when any one zone reaches the
  parameter set's voltage cut-offs.
* `mesh_h` — target element size of each zone's mesh, as for
  `pybamm.ScikitFemGenerator3D`.
* `options` — model options, given to every zone. `"thermal"` must be
  `"lumped"`, `"dimensionality"` 0, and `"cell geometry"` `"pouch"`: the stack's
  fields are the thermal model, and each zone keeps a lumped temperature for
  them to replace. `"surface temperature"` is recorded on the stack's `options`
  only, since the zones have no casing; the stack reports its own face
  temperatures either way.
* `zone_model` — `zone_model(options)` returning one zone's built model, in
  place of `ZONE_MODEL(options=options)`: for example a model built with
  `build=False` whose thermal submodel is replaced before it is built. It must
  honour the options it is handed and keep a lumped temperature.
* `coating` — `"double-sided"` (default) or `"single-sided"`: whether neighbouring
  unit cells share their current collector foils. See "Unit cells and foils".

For example, a silicon-graphite negative electrode with hysteresis on the
silicon and Marcus-Hush-Chidsey kinetics:

```python
model = MultiLayer3DThermalSPMe(
    num_physical_layers=24,
    num_subdivisions=4,
    options={
        "particle phases": ("2", "1"),
        "open-circuit potential": (("single", "one-state hysteresis"), "single"),
        "intercalation kinetics": ("Marcus-Hush-Chidsey", "symmetric Butler-Volmer"),
    },
)
```

### Parameters

The parameter set describes one unit cell, and `"Current function [A]"` is the
current through the whole stack. `apply_stack_scaling` multiplies `"Nominal cell
capacity [A.h]"` by the unit cells that share that current — all of them in
parallel, one zone's in series — so that an experiment's C-rate is each unit
cell's own. Leave `"Number of electrodes connected in parallel to make a cell"`
at 1: the stack's unit cells are this model's layers, and setting it too would
divide the current twice.

#### Unit cells and foils

`"Negative current collector thickness [m]"` and `"Positive current collector
thickness [m]"` are always the thickness of a whole foil, as in PyBaMM's own
parameter sets, whatever the coating.

With `coating="double-sided"`, the default, each foil is coated on both faces and
serves the unit cells on either side of it, so the stack alternates direction and
every unit cell owns half of each foil:

```
          unit cell k                unit cell k+1
   |<------------------------->|<------------------------->|
   ½Cu |  n  |  s  |  p  | ½Al ½Al |  p  |  s  |  n  | ½Cu ...
```

One unit cell is then `L_n + L_s + L_p + (L_cc,n + L_cc,p) / 2` thick
(`unit_cell_thickness`), and the stack `num_physical_layers` times that. With
`coating="single-sided"`, every unit cell has whole foils of its own, as PyBaMM's
single-cell models assume, and is `L_n + L_s + L_p + L_cc,n + L_cc,p` thick.

Double-sided is the default because it is how stacked pouch cells are built:
every inner electrode is coated on both faces of its foil. Counting a whole foil
per unit cell would make such a stack too thick, give it too much foil mass and
heat capacity, and overstate its in-plane conductivity, by a share that grows
with the foils' fraction of the unit cell (12% of the thickness for 6 µm copper
and 15 µm aluminium foils under 80 µm of electrodes and separator). Single-sided
is kept for cells whose electrodes are coated on one face only, and to compare
with PyBaMM's single-cell models, which carry whole foils, under the same
parameter set.

A unit cell's foil share enters everything that depends on the foils' thickness:
the stack's thickness, so the volume each zone's heat is spread over; the
thickness-weighted heat capacity and in-plane thermal conductivity; and the
series conduction through the stack. With `"use lumped thermal capacity"`,
`"Cell heat capacity [J.K-1.m-3]"` is per volume of this unit cell. The
electrochemistry does not depend on the foils' thickness. The two outermost foils
serve a single unit cell, so a double-sided stack is half a foil short at each
outer face.

`default_parameter_values` and `apply_stack_scaling` add these where the
parameter set lacks them:

* `"Inter-layer thermal contact resistance [K.m2.W-1]"`, default `0`: contact
  resistance between adjacent zones, an adhesive or gas gap, on top of
  conduction through the zones themselves. Each zone's field conducts with
  `lambda_eff`, the in-plane mean, so the interface between two zones carries
  their series conduction, `zone_series_resistance(T)`: the unit cell's layers,
  each foil at the unit cell's share of it, as thermal resistances in series,
  times the unit cells in a zone. The two outer zones carry half a zone of it in series with
  the cooling of the stack's outer faces. The layers' thicknesses and
  thermal conductivities therefore set the through-stack conductivity, and the
  number of unit cells the stack's thickness: a stack of thicker electrodes, or
  more of them, holds a larger core-to-skin difference. The metal foils are a
  negligible part of it (0.06% on `Marquis2019` with whole foils).
* `"<Face> face heat transfer coefficient [W.m-2.K-1]"` for the `Left` (`x = 0`,
  zone 0's outer face), `Right`, `Front`, `Back`, `Bottom`, and `Top` faces,
  default `10`.

## Variables

Per zone `i`, each prefixed `"Layer i "`:

* `temperature [K]` — the 3D field — and `average temperature [K]`.
* `heat generation [W.m-3]` and `heat capacity [J.K-1.m-3]`, per unit volume of
  a unit cell, current collectors included.
* `voltage [V]`, `current [A]` (the zone's), `per-unit-cell current [A]`, and,
  in parallel, `current fraction`, which is undefined at rest.
* `Total current density [A.m-2]`, PyBaMM's per-unit-cell current density, and
  each electrode's `X-averaged ... interfacial current density [A.m-2]`.
* A selection of the zone model's own variables under their PyBaMM names, for
  one unit cell: the `X-averaged ` and `Volume-averaged ` concentrations,
  stoichiometries, open-circuit potentials, hysteresis states, overpotentials,
  ohmic losses and heat terms; `Surface open-circuit voltage [V]`,
  `Discharge capacity [A.h]`, `Total lithium in electrolyte [mol]`, the heat
  terms in watts (`Total heating [W]`, `Reversible heating [W]`, ...), and, where
  the model has them, the electrolyte concentration and potential and the
  electrode potentials.

For the stack: `Voltage [V]`, `Current [A]`, `Total current density [A.m-2]`
(the mean over its unit cells), `Volume-averaged total heating [W.m-3]`, the heat
terms in watts and `Discharge capacity [A.h]` summed over every unit cell, and
its temperatures:

* `Left face temperature [K]` and `Right face temperature [K]` — the stack's two
  outer faces (zone 0's `x_min` and the last zone's `x_max`), each averaged over
  the footprint and named as its heat transfer coefficient is, so that under
  one-sided cooling it is clear which face is which.
* `Surface temperature [K]` — the mean of the two outer faces, where a skin
  thermocouple sits.
* `Core temperature [K]` — the stack's mid-plane, averaged over the footprint:
  the mean of the two faces at the middle interface for an even number of
  zones, the middle zone's average for an odd number.
* `Core-to-skin temperature difference [K]` — core minus surface.
* `Stack-averaged temperature [K]` — the volume average (also as
  `Volume-averaged cell temperature [K]` and `X-averaged cell temperature [K]`).
* `Maximum layer-averaged temperature [K]`, `Minimum layer-averaged temperature
  [K]`, and `Temperature spread [K]`, their difference.

## Validation

All of these run in `tests/test_multilayer_3d_thermal.py`, with `Marquis2019`
unless stated:

* Held isothermal by cooling every face at `1e4 W.m-2.K-1`, a symmetric two-zone
  stack matches `pybamm.lithium_ion.SPM`, `SPMe`, and `DFN` to within 1 mV over a
  30 minute discharge (measured 0.013, 0.008, and 0.005 mV).
* With a two-phase negative electrode and one-state hysteresis on its secondary
  phase (`Chen2020_composite`), with Butler-Volmer or Marcus-Hush-Chidsey kinetics
  on the negative electrode, the SPM, SPMe and DFN stacks match PyBaMM's own
  models under the same options to within 1 mV.
* PyBaMM's own `"heat of mixing"`, where it builds (one particle phase), reaches
  every zone, and an insulated stack heats as a lumped cell with the same option.
* With `"use lumped thermal capacity"`, each zone's heat capacity is `"Cell heat
  capacity [J.K-1.m-3]"`, and an insulated stack heats as a lumped cell with it.
* `"surface temperature": "lumped"` is kept on the stack's options, and the zones
  stay without a casing.
* A `zone_model` that swaps a submodel before building gives the same solution
  when the swapped-in submodel is PyBaMM's own.
* An experiment discharges to its own voltage cut-off, below the parameter set's,
  in parallel and in series: no zone keeps a voltage limit the experiment does
  not relax.
* In series, cooled through one face, the cold zone reaching the lower or upper
  cut-off stops the stack, with the warm zone still inside its limits.
* Insulated on every face, the stack heats as PyBaMM's lumped model of one unit
  cell to `rtol=1e-3` (measured 1e-5 and better): the same heat, over the same
  volume, current collectors included, and the same heat capacity. A
  double-sided stack matches a lumped cell given half of each foil; a
  single-sided one, a lumped cell with whole foils.
* A unit cell is its electrodes and separator plus half of each foil when
  double-sided, or whole foils when single-sided, in the stack's geometry and in
  its series conduction.
* Insulated on every face, the stack stores the heat it generates to within 0.1%
  (measured 0.002%). This pins the interface coupling: heat leaves one zone
  exactly as it enters the next.
* The stack's heat terms in watts are its zones' times the unit cells in each.
* The SPMe and DFN electrolytes conserve lithium to `rtol=1e-9`.
* A uniform parallel stack starts from rest with no current in any zone, then
  shares the load.
* After a discharge cooled through one face, the zones balance at rest: the
  cold zone, which carried less of the discharge, gives current to the warm one,
  and the zone currents sum to zero.
* PyBaMM's `"contact resistance"` option drops each unit cell's voltage by its
  own current times `"Contact resistance [Ohm]"`.
* Cooled through both big faces, core-to-skin is the same in 2 zones as in 12,
  and within 5% of a uniformly heated slab conducting in series.
* Cooled alike on both big faces, with odd and even numbers of zones, the two
  face temperatures agree, the surface sits below the stack average and the core
  above it. Cooled on the left face only, the left face is the colder.
* The interface resistance is the unit cell's layers in series, far below the
  in-plane `lambda_eff`.
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
* Inside a zone the field conducts with `lambda_eff`, the in-plane mean, in
  every direction, so a zone is close to isothermal through its own thickness;
  the series conduction through the stack sits on the interfaces between zone
  centres and at the outer faces. More zones resolve the shape of the profile
  and a non-uniform heat source, not its core-to-skin difference.
* A zone's electrochemistry sees only its own volume-averaged temperature, so the
  field's in-plane gradients carry heat but do not feed back into the kinetics.
* Each zone generates its heat uniformly over its volume.
* `"heat of mixing": "true"` is PyBaMM's own term, which does not build on an
  electrode with more than one particle phase; a zone model with its own thermal
  submodel can be passed as `zone_model` until it does.
* Tabs, current collector resistance beyond PyBaMM's `"contact resistance"`
  option, and in-plane current distribution within a layer are not modelled.

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
