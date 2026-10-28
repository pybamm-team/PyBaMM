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

$$
\frac{\partial c_\mathrm{vol}}{\partial t} = -\frac{a j}{\varepsilon_s F},
$$

where $c_\mathrm{vol}$ is the volume-averaged AM concentration, $a$ the specific surface
area, $\varepsilon_s$ the AM volume fraction and $j$ the reaction rate. The
surface-averaged concentration $c_\mathrm{surf}$ used in the kinetics is given by one of
four definitions (Table I of Part I, and Part II for DC1t):

| `"model type"` | surface concentration |
| --- | --- |
| `"DC1"` (default) | $c_\mathrm{surf} = c_\mathrm{vol} + \langle s \rangle_A j$ |
| `"DC1t"` | $c_\mathrm{surf} = c_\mathrm{vol} + \langle s^t \rangle_A \star \partial_t j$ (transient closure, see below) |
| `"DC0"` | $c_\mathrm{surf} = c_\mathrm{vol}$ |
| `"Yang"` | $c_\mathrm{surf} = c_\mathrm{vol} + \gamma j$, with $\gamma = \left(l_d^2/(6 l_k^2) - l_d/(2 l_k)\right) l_k/(D F)$ |

In DC1, $\langle s \rangle_A$ [mol.m-1.A-1] is the **closure variable**: the surface average
of the solution of a closure problem solved once on the electrode
microstructure (e.g. with the `solveclosure` tool, ref. 56 of the paper). It
depends only on the microstructure, not on operating conditions or material
properties (when non-dimensionalised). No assumption on particle shape is made.
With `"calculate surface concentration a priori": "true"`, $j$ is replaced by
its electrode average, which removes one algebraic equation per mesh cell. This
is only appropriate for galvanostatic (constant-current) operation; see the DC
options below.

Yang and Tartakovsky's model (Yang and Tartakovsky, 2025) uses a boundary layer of thickness
$l_d = a_d \sqrt{D t}$ (up to $l_k$) inside particles of length scale $l_k$,
taken from the particle radius parameter, with a fitting parameter $a_d$.

The DC model is a fully macroscale description: it has no radial dimension.
Radial output variables (e.g. `"Positive particle concentration [mol.m-3]"`)
are the volume average broadcast in r; use
`"Positive particle surface concentration [mol.m-3]"` for the surface
concentration.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo

DualContinuum = zoo.load("DualContinuum")

# Defaults: PyBaMM's DFN parameter set plus the DC parameters below
solution = pybamm.Simulation(DualContinuum()).solve([0, 3600])

# Cathode half cell
model = DualContinuum({"working electrode": "positive"})

# LG M50 cell with the parameters of Paten et al. (2026), Table IV
solution = pybamm.Simulation(
    DualContinuum(), parameter_values=pybamm.ParameterValues("Paten2026_DC")
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

- `"closure variable"` sets how $\langle s \rangle_A$ is obtained:
  - `"dimensional"`: read from the parameter set;
  - `"dimensionless"`: $s^\ast$ is read from the parameter set and
    $\langle s \rangle_A = s^\ast L / (D F)$, with $L$ the electrode thickness;
  - `"isolated sphere"`: $\langle s \rangle_A = -R_\mathrm{eff} / (5 D F)$, the analytical solution
    of the closure problem for an isolated sphere of radius $R_\mathrm{eff}$. With it,
    DC1 coincides with PyBaMM's `"quadratic profile"` particle for spherical
    geometry.

  $D$ is evaluated at the local volume-averaged concentration. For DC1t the
  option also sets how the mode time constants are given; see the
  *Transient closure (DC1t)* section below.
- `"calculate surface concentration a priori": "true"` estimates $j$ from the
  applied current, so it is only for galvanostatic (constant-current)
  operation. The model raises an error with any `"operating mode"` other than
  `"current"`. It cannot detect voltage, power or CV steps inside a
  `pybamm.Experiment`, so do not combine the option with such steps.
- `"surface area"`: with `"from image"` (the default), the specific surface area
  is built from areas measured on images of the microstructure; see
  *Specific surface area from images* below. `"spherical"` uses
  $a = 3 \varepsilon_s / R$, as in the DFN.
- `"lithium foil surface porosity": "true"` multiplies the exchange-current
  density of the lithium foil by `"Separator surface porosity"`. Some DNS
  codes include the separator surface porosity in the kinetics at the foil,
  since the foil only reacts where it meets the separator's pores; turn this on
  when comparing with such a DNS. Leave it off otherwise: a foil
  exchange-current density taken from the literature is usually given per
  geometric area, so it already includes the contact with the separator.

### Transient closure (DC1t)

In DC1 the closure variable is its steady value. In the transient model of
Part II (Paten et al., 2026b) it depends on time, $\langle s^t \rangle_A(t)$, starting from
0 and reaching the steady value $s_0$, and the surface concentration involves a
time convolution with the history of the reaction rate:

$$
c_\mathrm{surf} = c_\mathrm{vol} + \langle s^t \rangle_A \star \frac{\partial j}{\partial t}.
$$

This model writes the closure variable as a sum of exponential modes,

$$
\langle s^t \rangle_A(t) = s_0 \sum_m w_m \left(1 - e^{-t/\theta_m}\right), \qquad \sum_m w_m = 1,
$$

with weights $w_m \ge 0$ and time constants $\theta_m$. For each mode the
convolution is then exactly one ODE,

$$
c_\mathrm{surf} = c_\mathrm{vol} + s_0 \sum_m w_m y_m, \qquad
\frac{\mathrm{d} y_m}{\mathrm{d} t} = \frac{j - y_m}{\theta_m}, \qquad y_m(0) = 0,
$$

where $y_m$ is an exponentially weighted average of the past reaction rate.
This holds for any current history, so constant-voltage holds, rests and drive
cycles need no special treatment, and the model runs directly from a
`pybamm.Experiment`. By default $j$ is the local reaction rate. With
`"calculate surface concentration a priori": "true"`, the modes are driven by
the electrode-averaged reaction rate instead, which reproduces the
piecewise-galvanostatic convolution of Part II (Eq. 25) and is only for
galvanostatic operation. Each mode adds one state per mesh cell
(`"Positive electrode transient closure mode m [A.m-2]"`); 8 modes on 20
cells is 160 states, against 400 for a DFN with 20 radial points.

The weights are normalised by their sum, so the closure variable always
starts at 0 and tends to $s_0$, and DC1t reduces to DC1 when all time
constants are short. For an isolated sphere the exact closure variable is a
sum of exponentials (Paten et al., 2026c), and DC1t with enough modes
reproduces the Fickian DFN.

How the modes are given depends on `"closure variable"`:

- `"dimensional"`: `"Positive electrode transient closure weight m"` and
  `"Positive electrode transient closure time constant m [s]"`, for
  $m = 1, \dots, N$, with $s_0$ in `"Positive electrode s0 surface average"`;
- `"dimensionless"`: the same weights, and time constants
  `"Positive electrode transient closure time constant m dimensionless"`,
  with $\theta_m = \theta_m^\ast L^2 / D$;
- `"isolated sphere"`: built-in modes fitted to the transient solution for an
  isolated sphere, scaled by $R_\mathrm{eff}^2 / D$; no parameters are needed.

$N$ is the `"transient closure modes"` option. The default parameter values
contain the isolated-sphere modes, so the default DC1t describes the same cell
as the Fickian DFN. With a concentration-dependent diffusivity, the
dimensionless and isolated-sphere time constants are scaled with
$D(c_\mathrm{vol})$, an approximation.

#### Choosing the modes: `fit_transient_closure`

The helper fits the modes to your closure variable. Give it the times `t` as a
NumPy array, and the closure variable `s` either as an array of its values at
those times (for example read from the output of a closure solve on an image)
or as a Python function of time, which is then evaluated at `t`:

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
neighbours), because each mode only shapes $s(t)$ within about a factor of
ten in time around its time constant, while $s(t)$ changes over several
factors of ten. By default they run from a tenth of the first data time to a
third of the time at which $s$ reaches 99% of $s_0$, and the number of modes is
increased until the largest error is below `tolerance` (default 0.5% of
$s_0$). The weights come from a non-negative least-squares fit with
$\sum_m w_m = 1$. As a rule of thumb, about two modes are needed for each
factor of ten between the shortest and the longest time constant: at the
default tolerance, the isolated sphere needs 8 modes when its time constants
span about four factors of ten (for example from 0.01 s to 100 s). Smoother
closure data can need fewer. Use
`n_modes`, `tolerance` or `time_constant_range` to change the choice, and
`dimensionless=True` for data scaled as $t D / L^2$ and $s D F / L$.

The helper checks the data and reports problems in `fit.warnings` (also
logged):

- **Unresolved early times.** At short times every geometry responds like a
  flat wall, $s \approx -2 \sqrt{t/D} / (\sqrt{\pi} F)$. Closure solves on
  voxel meshes that report cell-centre values give early data far beyond this
  limit (offset by about $-h/(2 D F)$, cell size $h$); the helper reports the
  implied $h$. Prefer a finer mesh near the surface, or use
  `fix_early_times=True`, which replaces the data before `resolved_after`
  (default $10 h^2/D$) by the short-time limit.
- **Overshoot.** The modes cannot represent $s(t)$ overshooting its final
  value, which usually means the closure solve is not converged.

`isolated_sphere_closure(t, radius, diffusivity)` gives the analytical
closure variable of an isolated sphere.

#### Example: tomographic NMC electrode

`pybamm.ParameterValues("Paten2026_3D_NMC")` is the NMC half cell reconstructed from a
tomographic image (no carbon-binder domain) used for the DNS comparisons of
Paten et al. (2026), and `parameter_sets.load_3d_case_data()` returns its
transient closure variable and a DNS of a discharge to 3.0 V followed by a
900 s rest (folder `data/3d_case_data_Paten_2026`).
`examples/fit_transient_closure.py` fits the modes and plots the models against
the DNS during the discharge and the rest:

| Voltage [V] after the end of discharge | +1 s | +10 s | +100 s | +900 s |
| --- | --- | --- | --- | --- |
| DNS | 3.225 | 3.409 | 3.569 | 3.599 |
| DFN | 3.242 | 3.367 | 3.429 | 3.429 |
| DC1 (steady closure) | 3.587 | 3.595 | 3.602 | 3.603 |
| DC1t (5 fitted modes) | 3.323 | 3.477 | 3.587 | 3.603 |

During the discharge DC1t is within 3 mV RMS of the DNS. The transient closure
captures most of the relaxation that the steady closure misses. Each model
stops at its own 3.0 V cut-off, and DC1t stops 1.1 s before the DNS; stopped at
the DNS's end time instead, it agrees within 0.1 mV after 900 s. The remaining
difference early in the rest (about 60 mV at +10 s at equal charge) is not
explained by the offset of the closure data, and may come from the nonlinear
kinetics and open-circuit potential near full lithiation.

### Specific surface area from images

With `"surface area": "from image"`, the reactive specific surface area of each
electrode is

$$
a = a_\text{AM-electrolyte} + \varepsilon^\text{surf}_\text{CBD} a_\text{AM-CBD}
+ \varepsilon^\text{surf}_\text{sep} a_\text{AM-separator},
$$

with each specific area (per unit electrode volume) measured on images of the
microstructure. The carbon-binder domain (CBD) and the separator are effective
media that contain electrolyte in their pores, so a reaction still takes place
at AM-CBD and AM-separator interfaces, albeit at a reduced rate. The surface
porosities $\varepsilon^\text{surf}_\text{CBD}$ and $\varepsilon^\text{surf}_\text{sep}$ express that reduction: they are
the fraction of those interfaces in contact with electrolyte. **We recommend
setting each surface porosity equal to the corresponding volume porosity.**

An effective particle radius $R_\mathrm{eff} = 3 \varepsilon_s / a$ is computed from the total
area and reported as `"Positive electrode effective particle radius [m]"`; it
is the radius used by the isolated-sphere closure. This is purely to simplify
compatibility with PyBaMM's DFN class, which this model builds on.

### SEI and other DFN options

The DC model accepts PyBaMM's `"SEI"` options and the DFN's other options, with
their parameters, for example

```python
model = DualContinuum(
    {
        "SEI": "reaction limited",
        "SEI film resistance": "distributed",
        "total interfacial current density as a state": "true",
    }
)
```

The last two options are what PyBaMM would otherwise derive from `"SEI"`;
giving them explicitly avoids relying on those derived defaults. With `"surface area": "from image"`, the SEI
grows on the same total specific surface area $a$ as the main reaction,
including the AM-CBD and AM-separator contributions. Particle-size
distributions and loss of active material are not supported.

### Composite electrodes (several particle phases)

An electrode with several active materials, e.g. graphite and silicon, uses
PyBaMM's `"particle phases"` option, which also requires a `"surface form"`:

```python
options = {"particle phases": ("2", "1"), "surface form": "algebraic"}
model = DualContinuum(options, {"model type": "DC1t"})
parameter_values = model.add_dc_parameters(pybamm.ParameterValues("Chen2020_composite"))
```

Each phase has its own volume-averaged concentration, closure variable and
modes, coupled to the other phases only through the electrolyte and the solid
potential, as in the DFN. As for PyBaMM's particle parameters, the DC
parameters of a composite electrode are prefixed by phase, e.g.
`"Primary: Negative electrode s0 surface average"` and `"Secondary: ..."`;
those of an electrode with a single phase are not.
`fit.parameter_values("Negative", phase="secondary")` writes the prefixed names.

`add_dc_parameters` adds the DC parameters a set lacks, for every phase, with
the DFN-equivalent values listed below; it never overwrites a value already in
the set. PyBaMM's default set has no composite parameters, so for a composite
model `default_parameter_values` is the DFN's default set alone.
`"calculate surface concentration a priori"` needs a single phase, since how
the current splits between phases is not known a priori.

### Parameters

| Parameter | Used when |
| --- | --- |
| `"{Domain} electrode s0 surface average"` [mol.m-1.A-1], the closure variable $\langle s \rangle_A$ | DC1, `"closure variable": "dimensional"` |
| `"{Domain} electrode s0 surface average dimensionless"` $s^\ast$, with $\langle s \rangle_A = s^\ast L / (D F)$ | DC1, `"closure variable": "dimensionless"` |
| `"{Domain} electrode specific surface area from image (AM-electrolyte) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-CBD) [m-1]"` | `"surface area": "from image"` |
| `"{Domain} electrode specific surface area from image (AM-separator) [m-1]"` | `"surface area": "from image"` |
| `"CBD surface porosity"`, `"Separator surface porosity"` | `"surface area": "from image"` |
| `"{Domain} electrode Yang fitting parameter"` $a_d$ | Yang |
| `"{Domain} electrode transient closure weight m"` $w_m$ | DC1t, `"dimensional"` or `"dimensionless"` |
| `"{Domain} electrode transient closure time constant m [s]"` $\theta_m$ | DC1t, `"dimensional"` |
| `"{Domain} electrode transient closure time constant m dimensionless"` $\theta_m^\ast$, with $\theta_m = \theta_m^\ast L^2/D$ | DC1t, `"dimensionless"` |

`default_parameter_values` is the DFN's default set (Marquis2019 for a full
cell, Xu2019 for a half cell) passed through `add_dc_parameters`, which adds:

- AM-electrolyte areas equal to the DFN's $3 \varepsilon_s / R$, no AM-CBD or
  AM-separator contact, a separator surface porosity equal to the separator
  porosity, and a CBD surface porosity of 0.5.
- The closure variable set to the isolated-sphere value $-R/(5 D F)$ at the
  reference state, so the defaults describe the same cell as the DFN's.
- A Yang fitting parameter of 1.
- For DC1t, the isolated-sphere modes (weights, and time constants scaled by
  $R^2 / D$ at the reference state).

`pybamm.ParameterValues("Paten2026_DC")` and `pybamm.ParameterValues("Paten2026_DFN")`
are the DC1 and DFN columns of Table IV (an LG M50 cell, fitted to the 1C
discharge of Chen et al. 2020), on top of PyBaMM's Chen2020 set for all other
values. The model zoo registers these names, and `"Paten2026_3D_NMC"`, as PyBaMM
parameter sets; the functions in `parameter_sets` return the same values as plain
dictionaries. The
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
- The image-based specific surface area and $R_\mathrm{eff}$ follow the formula above,
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
- Composite electrodes: splitting the graphite of Chen2020_composite into two
  identical phases reproduces the single-phase model (rtol 1e-7) for DC1,
  DC1t and Yang; for graphite and silicon, DC1t with the isolated-sphere modes
  of each phase reproduces PyBaMM's two-phase Fickian DFN within 1 mV.
- DC1t's dimensional, dimensionless and isolated-sphere inputs agree; DC1t
  tends to DC1 as the time constants go to zero; with the electrode-averaged
  reaction rate it equals the galvanostatic convolution $c_\mathrm{vol} + \langle s^t \rangle_A(t) j$.
- On the tomographic NMC electrode, DC1t with modes fitted to the closure data
  matches the DNS within 3 mV RMS during discharge, and within 80, 25 and 5 mV
  at 10, 100 and 900 s into the rest.
- `fit_transient_closure` recovers known modes, meets its tolerance, and
  flags unresolved early times and overshoot.

## Citation

See `CITATION.bib`. Please cite Paten2026 when using this model, Paten2026b
(Part II) when using DC1t, and Yang2025 when using the Yang model; they are
registered automatically, so `pybamm.print_citations()` credits them after use.
Paten2026c shows the formal equivalence between DC1t and the DFN for isolated
spheres.

## Maintainer

Isaac Paten (@isaacbasil) — tier: community
