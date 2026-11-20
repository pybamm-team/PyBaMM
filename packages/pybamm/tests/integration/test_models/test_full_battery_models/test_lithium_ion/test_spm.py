#
# Tests for the lithium-ion SPM model
#
import numpy as np
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPM

    @pytest.mark.parametrize("dimensionality", [1, 2])
    def test_potential_pair_parallel_electrodes(self, dimensionality):
        # N identical electrodes in parallel at current I must behave like a single
        # electrode at I / N
        model = self.model(
            {"current collector": "potential pair", "dimensionality": dimensionality}
        )
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        n_parallel, current = 10, 1

        solutions = []
        for n, I in [(1, current / n_parallel), (n_parallel, current)]:
            param = model.default_parameter_values
            param.update(
                {
                    "Number of electrodes connected in parallel to make a cell": n,
                    "Current function [A]": I,
                }
            )
            sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts)
            solutions.append(sim.solve([0, 600]))

        t = np.linspace(0, 600, 11)
        for name in ["Current collector current density [A.m-2]", "Voltage [V]"]:
            np.testing.assert_allclose(
                solutions[0][name](t), solutions[1][name](t), rtol=1e-3
            )

    def test_composite_psd_hysteresis_states(self):
        options = {
            "particle phases": ("2", "1"),
            "particle size": "distribution",
            "open-circuit potential": (
                ("one-state hysteresis", "one-state hysteresis"),
                "single",
            ),
        }
        model = self.model(options)
        parameter_values = pybamm.get_size_distribution_parameters(
            pybamm.ParameterValues("Chen2020_composite"), composite="negative"
        )
        primary_ocp = parameter_values["Primary: Negative electrode OCP [V]"]
        secondary_ocp = parameter_values["Secondary: Negative electrode OCP [V]"]
        parameter_values.update(
            {
                "Primary: Negative electrode lithiation OCP [V]": primary_ocp,
                "Primary: Negative electrode delithiation OCP [V]": primary_ocp,
                "Secondary: Negative electrode lithiation OCP [V]": secondary_ocp,
                "Secondary: Negative electrode delithiation OCP [V]": secondary_ocp,
                "Primary: Negative particle lithiation hysteresis decay rate": 1,
                "Primary: Negative particle delithiation hysteresis decay rate": 1,
                "Secondary: Negative particle lithiation hysteresis decay rate": 1,
                "Secondary: Negative particle delithiation hysteresis decay rate": 1,
                "Primary: Initial hysteresis state in negative electrode": 0,
                "Secondary: Initial hysteresis state in negative electrode": 0,
            },
            check_already_exists=False,
        )

        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            [0, 1]
        )

        for phase in ["primary", "secondary"]:
            (state,) = [
                variable
                for variable in model.rhs
                if variable.name
                == f"X-averaged negative electrode {phase} hysteresis state distribution"
            ]
            assert state.domain == [f"negative {phase} particle size"]
            state_at_end = solution[
                f"X-averaged negative electrode {phase} hysteresis state distribution"
            ](1)
            assert state_at_end.size > 1
            assert np.all(np.isfinite(state_at_end))
