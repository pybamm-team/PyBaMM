#
# Test some experiments
#
import numpy as np

import pybamm


class TestExperiments:
    def test_discharge_rest_charge(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/2 for 1 hour",
                "Rest for 1 hour",
                "Charge at C/2 for 1 hour",
            ],
            period="0.5 hours",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()
        np.testing.assert_allclose(
            sim._solution["Time [h]"].entries,
            np.array([0, 0.5, 1, 1 + 1e-9, 1.5, 2, 2 + 1e-9, 2.5, 3]),
            rtol=1e-7,
            atol=1e-6,
        )
        cap = model.default_parameter_values["Nominal cell capacity [A.h]"]
        np.testing.assert_allclose(
            sim._solution["Current [A]"].entries,
            [cap / 2] * 3 + [0] * 3 + [-cap / 2] * 3,
            rtol=1e-7,
            atol=1e-6,
        )

    def test_rest_discharge_rest(self):
        # An experiment which requires recomputing consistent states
        experiment = pybamm.Experiment(
            ["Rest for 5 minutes", "Discharge at 0.1C until 3V", "Rest for 30 minutes"],
            period="1 minute",
        )
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        sol = sim.solve()
        np.testing.assert_allclose(sol["Current [A]"].data[:5], 0, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(
            sol["Current [A]"].data[-29:], 0, rtol=1e-7, atol=1e-6
        )

    def test_gitt(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 10, period="6 minutes"
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()
        cap = model.default_parameter_values["Nominal cell capacity [A.h]"]
        np.testing.assert_allclose(
            sim._solution["Current [A]"].entries,
            [cap / 20] * 11 + [0] * 11 + ([cap / 20] * 11 + [0] * 11) * 9,
            rtol=1e-7,
            atol=1e-6,
        )

    def test_infeasible(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at 1C for 0.5 hours",),
            ]
            * 4
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        # this experiment fails during the third cycle (i.e. is infeasible)
        assert len(sol.cycles) == 3

    def test_drive_cycle(self):
        drive_cycle = np.array([np.arange(100), 5 * np.ones(100)]).T
        c_step = pybamm.step.current(
            value=drive_cycle, duration=100, termination=["< 4.00 V"]
        )
        experiment = pybamm.Experiment(
            [
                c_step,
            ],
        )
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
        )
        sol = sim.solve()
        # Add a small tolerance to account for numerical errors
        V_max = 4.00 - 1e-4
        assert np.all(sol["Terminal voltage [V]"].entries >= V_max)
