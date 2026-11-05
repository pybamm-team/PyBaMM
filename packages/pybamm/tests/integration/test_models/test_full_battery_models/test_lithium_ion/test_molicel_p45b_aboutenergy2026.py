#
# Integration tests for the Molicel P45B About:Energy (2026) parameter set
#

import pytest

import pybamm


class TestMolicelP45B_AboutEnergy2026:
    def test_discharge_capacity(self):
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "open-circuit potential": (
                    ("single", "one-state hysteresis"),
                    "single",
                ),
                "contact resistance": "true",
            }
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=pybamm.ParameterValues("MolicelP45B_AboutEnergy2026"),
            experiment=pybamm.Experiment(["Discharge at C/5 until 2.5 V"]),
        )
        solution = sim.solve()
        capacity = solution["Discharge capacity [A.h]"](solution.t[-1])
        assert capacity == pytest.approx(4.4945, rel=1e-3)
