"""Physics tests for the reference model zoo entry.

These are the contributor's own tests (the zoo's "Layer B"): the contract suite
already checks that the model imports, is well posed, builds, and solves, so
these pin *physical* results instead.
"""

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo

SOLVE_TIME = 1800
SERIES_RESISTANCE = 0.05


def solve(series_resistance):
    model = zoo.load("SPMSeriesResistance")()
    parameter_values = model.default_parameter_values
    parameter_values["Series resistance [Ohm]"] = series_resistance
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    return simulation.solve([0, SOLVE_TIME])


class TestSPMSeriesResistance:
    def test_default_parameter_values_carry_the_resistance(self):
        model = zoo.load("SPMSeriesResistance")()
        assert "Series resistance [Ohm]" in model.default_parameter_values

    def test_reduces_to_spm_at_zero_resistance(self):
        core = pybamm.lithium_ion.SPM()
        reference = pybamm.Simulation(core).solve([0, SOLVE_TIME])
        solution = solve(0.0)
        times = np.linspace(0, SOLVE_TIME, 50)
        np.testing.assert_allclose(
            solution["Voltage [V]"](times),
            reference["Voltage [V]"](times),
            rtol=1e-6,
        )

    def test_voltage_offset_equals_current_times_resistance(self):
        without = solve(0.0)
        with_resistance = solve(SERIES_RESISTANCE)
        times = np.linspace(0, SOLVE_TIME, 50)
        current = without["Current [A]"](times)
        np.testing.assert_allclose(
            without["Voltage [V]"](times) - with_resistance["Voltage [V]"](times),
            current * SERIES_RESISTANCE,
            rtol=1e-5,
        )

    def test_overpotential_variable_matches_the_drop(self):
        solution = solve(SERIES_RESISTANCE)
        times = np.linspace(0, SOLVE_TIME, 50)
        np.testing.assert_allclose(
            solution["Series resistance overpotential [V]"](times),
            -solution["Current [A]"](times) * SERIES_RESISTANCE,
            rtol=1e-12,
        )

    def test_voltage_as_a_state_is_rejected(self):
        with pytest.raises(pybamm.OptionError, match=r"voltage as a state"):
            zoo.load("SPMSeriesResistance")({"voltage as a state": "true"})
