#
# Tests for the positive electrode degradation option
#
import numpy as np
import pytest

import pybamm


class TestPositiveElectrodeDegradation:
    @pytest.mark.parametrize("model", [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN])
    def test_shell_grows_and_lithium_is_conserved(self, model):
        model = model({"positive electrode degradation": "true"})
        experiment = pybamm.Experiment(
            [
                (
                    "Charge at 1 C until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Discharge at 1 C until 2.8 V",
                )
            ]
            * 2
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=pybamm.ParameterValues("Zhuo2023"),
            experiment=experiment,
        )
        solution = sim.solve(calc_esoh=False)
        t = solution.t

        boundary = np.ravel(solution["X-averaged moving phase boundary location"](t))
        assert boundary[-1] < boundary[0]
        np.testing.assert_array_less(np.diff(boundary), 1e-9)

        shell = np.ravel(
            solution["X-averaged positive particle shell volume fraction"](t)
        )
        np.testing.assert_allclose(shell, 1 - boundary**3, rtol=0, atol=1e-5)

        discharge_capacities = []
        for cycle in solution.cycles:
            discharge = cycle.steps[2]
            capacity = discharge["Discharge capacity [A.h]"]
            discharge_capacities.append(
                float(capacity(discharge.t[-1]) - capacity(discharge.t[0]))
            )
        assert discharge_capacities[1] < discharge_capacities[0]

        lithium = np.ravel(solution["Total lithium in particles [mol]"](t))
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-4)
