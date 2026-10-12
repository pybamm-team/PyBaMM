# DualContinuum

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/dual_continuum.json)

## Summary

Dual-continuum (DC) model of a lithium-ion cell (Paten et al., 2026), built as a
subclass of `pybamm.lithium_ion.DFN`. The electrolyte, solid-phase conduction,
kinetics, thermal and lithium-metal counter-electrode physics are identical to
those in PyBaMM's DFN. The difference is the active material (AM): instead of
solving radial diffusion in idealised spherical particles at every point of the
electrode, the AM mass balance is homogenised with the volume-averaging
technique,

    d c_vol / dt = - a j / (eps_s F),

where `c_vol` is the volume-averaged AM concentration, `a` the specific surface
area, `eps_s` the AM volume fraction and `j` the reaction rate. The
surface-averaged concentration `c_surf` used in the kinetics is given by one of
four definitions (Table I of Part I, and Part II for DC1t):

| `"model type"` | surface concentration |
| --- | --- |
| `"DC1"` (default) | `c_surf = c_vol + <s>_A j` |
| `"DC1t"` | `c_surf = c_vol + <s^t>_A ⋆ ∂j/∂t` (transient closure, see below) |
| `"DC0"` | `c_surf = c_vol` |
| `"Yang"` | `c_surf = c_vol + gamma j`, `gamma = (l_d^2/(6 l_k^2) - l_d/(2 l_k)) l_k/(D F)` |

In DC1, `<s>_A` [mol.m-1.A-1] is the **closure variable**: the surface average
of the solution of a closure problem solved once on the electrode
microstructure (e.g. with the `solveclosure` tool, ref. 56 of the paper). It
depends only on the microstructure, not on operating conditions or material
properties (when non-dimensionalised). No assumption on particle shape is made.
With `"calculate surface concentration a priori": "true"`, `j` is replaced by
its electrode average, which removes one algebraic equation per mesh cell. This
is only appropriate for galvanostatic (constant-current) operation; see the DC
options below.

Yang and Tartakovsky's model uses a boundary layer of thickness
`l_d = a_d sqrt(D t)` (up to `l_k`) inside particles of length scale `l_k`,
taken from the particle radius parameter, with a fitting parameter `a_d`.

The DC model is a fully macroscale description: it has no radial dimension.
Radial output variables (e.g. `"Positive particle concentration [mol.m-3]"`)
are the volume average broadcast in r; use
`"Positive particle surface concentration [mol.m-3]"` for the surface
concentration.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import parameter_sets

DualContinuum = zoo.load("DualContinuum")

# Defaults: PyBaMM's DFN parameter set plus the DC parameters below
solution = pybamm.Simulation(DualContinuum()).solve([0, 3600])

# Cathode half cell
model = DualContinuum({"working electrode": "positive"})

# LG M50 cell with the parameters of Paten et al. (2026), Table IV
solution = pybamm.Simulation(
    DualContinuum(), parameter_values=parameter_sets.paten2026_dc1()
).solve([0, 3600])

# Any parameter set written for the DFN (no DC parameters needed)
model = DualContinuum(
    dc_options={"closure variable": "isolated sphere", "surface area": "spherical"}
)
solution = pybamm.Simulation(
    model, parameter_values=pybamm.ParameterValues("Chen2020")
).solve([0, 3600])
```

### DC options

| Option | Values (first is default) |
| --- | --- |
| `"model type"` | `"DC1"`, `"DC1t"`, `"DC0"`, `"Yang"` |
| `"transient closure modes"` | a positive integer, default 8 (DC1t only) |
| `"calculate surface concentration a priori"` | `"false"`, `"true"` (DC1 and DC1t) |
| `"closure variable"` | `"dimensional"`, `"dimensionless"`, `"isolated sphere"` (DC1 and DC1t) |
| `"surface area"` | `"from image"`, `"spherical"` |
| `"lithium foil surface porosity"` | `"false"`, `"true"` (half cells only) |

- `"closure variable"` sets how `<s>_A` is obtained:
  - `"dimensional"`: read from the parameter set;
  - `"dimensionless"`: `s*` is read from the parameter set and
    `<s>_A = s* L / (D F)`, with `L` the electrode thickness;
  - `"isolated sphere"`: `<s>_A = -R_eff / (5 D F)`, the analytical solution
    of the closure problem for an isolated sphere of radius `R_eff`. With it,
    DC1 coincides with PyBaMM's `"quadratic profile"` particle for spherical
    geometry.

  `D` is evaluated at the local volume-averaged concentration. For DC1t the
  option also sets how the mode time constants are given; see
  [Transient closure (DC1t)](#transient-closure-dc1t).
- `"calculate surface concentration a priori": "true"` estimates `j` from the
  applied current, so it is only for galvanostatic (constant-current)
  operation. The model raises an error with any `"operating mode"` other than
  `"current"`. It cannot detect voltage, power or CV steps inside a
  `pybamm.Experiment`, so do not combine the option with such steps.
- `"surface area": "spherical"` uses `a = 3 eps_s / R`, as in the DFN.
- `"lithium foil surface porosity": "true"` multiplies the exchange-current
  density of the lithium foil by `"Separator surface porosity"`. Some DNS
  codes include the separator surface porosity in the kinetics at the foil,
  since the foil only reacts where it meets the separator's pores; turn this on
  when comparing with such a DNS. Leave it off otherwise: a foil
  exchange-current density taken from the literature is usually given per
  geometric area, so it already includes the contact with the separator.

### Transient closure (DC1t)

In DC1 the closure variable is its steady value. In the transient model of
Part II (Paten et al., 2026b) it depends on time, `<s^t>_A(t)`, starting from
0 and reaching the steady value `s0`, and the surface concentration involves a
time convolution with the history of the reaction rate:

    c_surf = c_vol + <s^t>_A ⋆ ∂j/∂t.

This model writes the closure variable as a sum of exponential modes,

    <s^t>_A(t) = s0 Σ_m w_m (1 - exp(-t / θ_m)),     Σ_m w_m = 1,

with weights `w_m ≥ 0` and time constants `θ_m`. For each mode the
convolution is then exactly one ODE,

    c_surf = c_vol + s0 Σ_m w_m y_m,     dy_m/dt = (j - y_m) / θ_m,  y_m(0) = 0,

where `y_m` is an exponentially weighted average of the past reaction rate.
This holds for any current history, so constant-voltage holds, rests and drive
cycles need no special treatment, and the model runs directly from a
`pybamm.Experiment`. By default `j` is the local reaction rate. With
`"calculate surface concentration a priori": "true"`, the modes are driven by
the electrode-averaged reaction rate instead, which reproduces the
piecewise-galvanostatic convolution of Part II (Eq. 25) and is only for
galvanostatic operation. Each mode adds one state per mesh cell
(`"Positive electrode transient closure mode m [A.m-2]"`); 8 modes on 20
cells is 160 states, against 400 for a DFN with 20 radial points.

The weights are normalised by their sum, so the closure variable always
starts at 0 and tends to `s0`, and DC1t reduces to DC1 when all time
constants are short. For an isolated sphere the exact closure variable is a
sum of exponentials (Paten et al., 2026c), and DC1t with enough modes
reproduces the Fickian DFN.

How the modes are given depends on `"closure variable"`:

- `"dimensional"`: `"Positive electrode transient closure weight m"` and
  `"Positive electrode transient closure time constant m [s]"`, for
  `m = 1, ..., N`, with `s0` in `"Positive electrode s0 surface average"`;
- `"dimensionless"`: the same weights, and time constants
  `"Positive electrode transient closure time constant m dimensionless"`,
  with `θ = θ* L² / D`;
- `"isolated sphere"`: built-in modes fitted to the transient solution for an
  isolated sphere, scaled by `R_eff² / D`; no parameters are needed.

`N` is the `"transient closure modes"` option. The default parameter values
contain the isolated-sphere modes, so the default DC1t describes the same cell
as the Fickian DFN.

#### Choosing the modes: `fit_transient_closure`

The helper fits the modes to your closure variable, given as arrays (for
example from a closure solve on an image) or as a function of time:

```python
import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import fit_transient_closure

fit = fit_transient_closure(t, s, diffusivity=4e-14)  # t [s], s [mol.m-1.A-1]
print(fit)  # modes, largest error and warnings
fit.plot()  # data and fit on a log time axis, with the error

model = zoo.load("DualContinuum")(
    {"working electrode": "positive"},
    {"model type": "DC1t", "transient closure modes": fit.n_modes},
)
parameter_values.update(fit.parameter_values("Positive"), check_already_exists=False)
```

The time constants are spaced logarithmically (a constant ratio between
neighbours), because each mode only shapes `s(t)` within about a factor of
ten in time around its time constant, while `s(t)` changes over several
factors of ten. By default they run from a tenth of the first data time to a
third of the time at which `s` reaches 99% of `s0`, and the number of modes is
increased until the largest error is below `tolerance` (default 0.5% of
`s0`). The weights come from a non-negative least-squares fit with
`Σ w_m = 1`. Two to three modes per factor of ten in time are typical. Use
`n_modes`, `tolerance` or `time_constant_range` to change the choice, and
`dimensionless=True` for data scaled as `t D / L²` and `s D F / L`.

The helper checks the data and reports problems in `fit.warnings` (also
logged):

- **Unresolved early times.** At short times every geometry responds like a
  flat wall, `s ≈ -(2/√π) √(t/D) / F`. Closure solves on voxel meshes often
  report a surface value offset by about `-h/(2DF)` (extrapolation from cell
  centres to the surface, cell size `h`), so the early data lies far beyond
  this limit; the helper reports the implied `h`. The fit then puts the
  offset in a fast mode, and the voltage jumps too far when the current
  changes. Prefer recomputing the surface average without the offset or on a
  finer mesh near the surface. As a workaround, `fix_early_times=True`
  drops the data before `resolved_after` (default `10 h²/D`), adds points of
  the short-time limit where it is valid, and fits across the gap.
- **Overshoot.** The modes cannot represent `s(t)` overshooting its final
  value, which usually means the closure solve is not converged.

`isolated_sphere_closure(t, radius, diffusivity)` gives the analytical
closure variable of an isolated sphere.

#### Example: tomographic NMC electrode

`parameter_sets.paten2026_3d_case()` is the NMC half cell reconstructed from a
tomographic image (no carbon-binder domain) used for the DNS comparisons of
Paten et al. (2026), and `parameter_sets.load_3d_case_data()` returns its
transient closure variable and a DNS of a discharge to 3.0 V followed by a
900 s rest (folder `data/3d_case_data_Paten_2026`).
`examples/fit_transient_closure.py` fits the modes and compares the models:

| Voltage [V] after the end of discharge | +1 s | +10 s | +100 s | +900 s |
| --- | --- | --- | --- | --- |
| DNS | 3.225 | 3.409 | 3.569 | 3.599 |
| DFN | 3.242 | 3.367 | 3.429 | 3.429 |
| DC1 (steady closure) | 3.587 | 3.595 | 3.602 | 3.603 |
| DC1t (5 fitted modes) | 3.323 | 3.477 | 3.587 | 3.603 |

During the discharge DC1t is within 3 mV RMS of the DNS. The transient closure
captures most of the relaxation that the steady closure misses. The remaining
early difference comes partly from the unresolved early times of this closure
data (the helper warns about them).

### Specific surface area from images

With `"surface area": "from image"`, the reactive specific surface area of each
electrode is

    a = a_AM-electrolyte + eps_CBD_surf a_AM-CBD + eps_sep_surf a_AM-separator,

with each specific area (per unit electrode volume) measured on images of the
microstructure. The carbon-binder domain (CBD) and the separator are effective
media that contain electrolyte in their pores, so a reaction still takes place
at AM-CBD and AM-separator interfaces, albeit at a reduced rate. The surface
porosities `eps_CBD_surf` and `eps_sep_surf` express that reduction: they are
the fraction of those interfaces in contact with electrolyte. **We recommend
setting each surface porosity equal to the corresponding volume porosity.**

An effective particle radius `R_eff = 3 eps_s / a` is computed from the total
area and reported as `"Positive electrode effective particle radius [m]"`; it
is the radius used by the isolated-sphere closure. This is purely to simplify
compatibility with PyBaMM's DFN class, which this model builds on.

### SEI and other DFN options

The DC model accepts PyBaMM's `"SEI"` options (e.g.
`DualContinuum({"SEI": "reaction limited"})`) and the DFN's other options, with
their parameters. With `"surface area": "from image"`, the SEI
grows on the same total specific surface area `a` as the main reaction,
including the AM-CBD and AM-separator contributions.

### Parameters

| Parameter | Used when |
| --- | --- |
| `"{Domain} electrode s0 surface average"` [mol.m-1.A-1], the closure variable `<s>_A` | DC1, `"closure variable": "dimensional"` |
| `"{Domain} electrode s0 surface average dimensionless"` `s*`, with `<s>_A = s* L / (D F)` | DC1, `"closure variable": "dimensionless"` |
| `"{Domain} electrode specific surface area from image (AM-electrolyte) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-CBD) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-separator) [m-1]"` | `"surface area": "from image"` |
| `"CBD surface porosity"`, `"Separator surface porosity"` | `"surface area": "from image"` |
| `"{Domain} electrode Yang fitting parameter"` `a_d` | Yang |
| `"{Domain} electrode transient closure weight m"` `w_m` | DC1t, `"dimensional"` or `"dimensionless"` |
| `"{Domain} electrode transient closure time constant m [s]"` `θ_m` | DC1t, `"dimensional"` |
| `"{Domain} electrode transient closure time constant m dimensionless"` `θ*_m`, with `θ = θ* L²/D` | DC1t, `"dimensionless"` |

`default_parameter_values` is the DFN's default set (Marquis2019 for a full
cell, Xu2019 for a half cell) plus:

- AM-electrolyte areas equal to the DFN's `3 eps_s / R`, no AM-CBD or
  AM-separator contact, separator surface porosity equal to the separator
  porosity, and a CBD surface porosity of 0.5;
- the closure variable set to the isolated-sphere value `-R/(5 D F)` at the
  reference state, so the defaults describe the same cell as the DFN's;
- a Yang fitting parameter of 1;
- for DC1t, the isolated-sphere modes (weights, and time constants scaled by
  `R² / D` at the reference state).

`parameter_sets.paten2026_dc1()` and `parameter_sets.paten2026_dfn()` return the
DC1 and DFN columns of Table IV (an LG M50 cell, fitted to the 1C discharge of
Chen et al. 2020), on top of PyBaMM's Chen2020 set for all other values. The
electrolyte tortuosities are given both as tortuosity factors and as the
equivalent Bruggeman coefficients, so either `"transport efficiency"` option
gives the same result.

## Validation

Pinned by `tests/test_dual_continuum.py`, for a full cell (Marquis2019) and a
cathode half cell (Xu2019):

- DC0 reproduces DFN `"uniform profile"` (rtol 1e-7).
- DC1 with the isolated-sphere closure variable reproduces DFN
  `"quadratic profile"` (rtol 1e-7), as do the default parameters, with
  dimensional and dimensionless closure variables.
- The image-based specific surface area and `R_eff` follow the formula above,
  and `"lithium foil surface porosity"` scales the foil exchange current by the
  separator surface porosity.
- Lithium in the active material is conserved to round-off.
- In a half cell with AM-CBD and AM-separator areas, surface porosities and
  the foil scaling, DC0, DC1 and DC1 a priori match reference voltages from
  the standalone implementation used in the paper to within 10 µV
  (`tests/data/standalone_reference.csv`, whose header records how it was
  generated).
- Splitting the specific surface area into AM-electrolyte, AM-CBD and
  AM-separator contributions gives the same result as the same total area.
- The Table IV parameter sets give identical results with the Bruggeman and
  tortuosity-factor transport options.
- With PyBaMM's `"SEI"` options, DC1 with the isolated-sphere closure variable
  reproduces the DFN with the `"quadratic profile"` particle (rtol 1e-7).
- DC1t with the default (isolated-sphere) modes reproduces the Fickian DFN
  with a fine radial mesh, within 0.5 mV for a constant-current discharge
  (full and half cell) and within 1 mV and 0.1 mA through a constant-current,
  constant-voltage and rest experiment.
- DC1t's dimensional, dimensionless and isolated-sphere inputs agree; DC1t
  tends to DC1 as the time constants go to zero; with the electrode-averaged
  reaction rate it equals the galvanostatic convolution `c_vol + <s^t>_A(t) j`.
- On the tomographic NMC electrode, DC1t with modes fitted to the closure data
  matches the DNS within 3 mV RMS during discharge, and within 80, 25 and 5 mV
  at 10, 100 and 900 s into the rest.
- `fit_transient_closure` recovers known modes, meets its tolerance, and
  flags unresolved early times and overshoot.

Not yet validated here: the Yang closure (only its dependence on the particle
radius is checked), and the experimental comparison of the paper. Not yet
ported from the standalone implementation: minimum plating overpotential
outputs. DC1t with a concentration-dependent diffusivity scales the time
constants with `D(c_vol)`, an approximation. Particle-size
distributions, multiple particle phases and loss of active material are not
supported.

## Citation

See `CITATION.bib`. Please cite Paten2026 when using this model, and
Paten2026b (Part II) when using DC1t; they are registered automatically, so
`pybamm.print_citations()` credits them after use. Paten2026c shows the formal
equivalence between DC1t and the DFN for isolated spheres.

## Maintainer

Isaac Paten (@isaacbasil) — tier: community
