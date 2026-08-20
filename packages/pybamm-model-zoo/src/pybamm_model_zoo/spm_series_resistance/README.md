# SPMSeriesResistance

![status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm-model-zoo/badges/spm_series_resistance.json)

## Summary

The Single Particle Model with a lumped ohmic series resistance: the terminal
voltage is the SPM voltage minus `I R`, where `R` is a new parameter,
`"Series resistance [Ohm]"`. It stands in for everything outside the
electrochemistry — tabs, welds, busbars, cabling — when a measured cell shows a
constant offset that the electrochemical model alone cannot account for. Prefer
it over post-processing the SPM voltage when the drop should also move the
voltage cut-offs, the reported power, and the ECM resistance, all of which the
model derives from the shifted voltage.

This is the model zoo's **reference entry**: it is deliberately the smallest thing
that is still a real model. Copy this folder as the starting point for your own.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo

model = zoo.load("SPMSeriesResistance")()
parameter_values = model.default_parameter_values
parameter_values["Series resistance [Ohm]"] = 0.05

simulation = pybamm.Simulation(model, parameter_values=parameter_values)
solution = simulation.solve([0, 1800])
print(solution["Voltage [V]"](900))
```

## Validation

* At `R = 0` the model reproduces `pybamm.lithium_ion.SPM` voltage to
  `rtol=1e-6` over a 1800 s 1C discharge.
* At `R > 0` under constant current, the voltage offset from the `R = 0` solution
  equals `I R` to `rtol=1e-5` — the residual is interpolation error between two
  independently adaptive solves, not a physical difference.
* Both checks run in `tests/test_spm_series_resistance.py`.

Not validated: any operating mode where the external circuit reads the terminal
voltage back, since the drop is applied after the circuit submodel has been
built. Under power or voltage control the *internal* voltage is what the circuit
holds, and the `"voltage as a state"` option raises `pybamm.OptionError` for the
same reason. Thermal coupling of the `I^2 R` loss is not included: the resistance
is outside the cell in this model, so its heat is not fed to the thermal
submodel.

## Citation

See `CITATION.bib`. Cite `PyBaMMModelZoo2026` for this entry and
`Marquis2019` for the underlying SPM; both are registered automatically, so
`pybamm.print_citations()` lists them after you use the model.

## Maintainer

The PyBaMM Team (@pybamm-team/maintainers) — tier: core
