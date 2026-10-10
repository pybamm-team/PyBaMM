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
