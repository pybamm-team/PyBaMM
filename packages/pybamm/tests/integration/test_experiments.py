#
# Test some experiments
#
import numpy as np
import pytest

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

    def test_drive_cycle_duration(self):
        # Test that the drive cycle is truncated to duration timespan
        tf = 1
        drive_cycle = np.column_stack(
            (
                np.array([0, tf + 1]),
                np.array([0, -1 / 100]),
            )
        )
        c_step = pybamm.step.current(value=drive_cycle, duration=tf)
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

        assert sol.t[-1] == tf


class TestExperimentSensitivities:
    """Sensitivities across experiment step changes against finite differences."""

    @staticmethod
    def _solve(model_options, name, value, sensitivities, continue_from_solution):
        def simulation(steps):
            parameter_values = pybamm.ParameterValues("Chen2020")
            parameter_values.update({name: "[input]"})
            return pybamm.Simulation(
                pybamm.lithium_ion.SPMe(model_options),
                parameter_values=parameter_values,
                experiment=pybamm.Experiment(steps, period="30 seconds"),
                var_pts={"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 8, "r_p": 8},
                solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12),
            )

        steps = [
            "Discharge at 1C for 5 minutes",
            "Rest for 2 minutes",
            "Charge at 0.5C for 2 minutes",
        ]
        kwargs = {"inputs": {name: value}, "calculate_sensitivities": sensitivities}
        if not continue_from_solution:
            return simulation(steps).solve(**kwargs)
        # A solution from another simulation is mapped by variable name
        starting_solution = simulation(steps[:1]).solve(**kwargs)
        return simulation(steps[1:]).solve(
            starting_solution=starting_solution, **kwargs
        )

    @pytest.mark.parametrize("continue_from_solution", [False, True])
    @pytest.mark.parametrize(
        "model_options, name, value",
        [
            # The surface potential differences are stored about a nonzero reference
            (
                {"surface form": "differential"},
                "Positive electrode conductivity [S.m-1]",
                0.18,
            ),
            # These inputs set the scale of the particle and electrolyte states
            ({}, "Maximum concentration in negative electrode [mol.m-3]", 33133.0),
            ({}, "Initial concentration in electrolyte [mol.m-3]", 1000.0),
        ],
        ids=["reference", "particle_scale", "electrolyte_scale"],
    )
    def test_sensitivities_across_model_changes(
        self, model_options, name, value, continue_from_solution
    ):
        solution = self._solve(
            model_options, name, value, [name], continue_from_solution
        )
        h = 1e-4 * value
        plus = self._solve(
            model_options, name, value + h, False, continue_from_solution
        )
        minus = self._solve(
            model_options, name, value - h, False, continue_from_solution
        )

        def steps(sol):
            return [step for cycle in sol.cycles for step in cycle.steps]

        assert len(steps(solution)) == 3
        finite_differences = [
            (step_plus["Voltage [V]"](step.t) - step_minus["Voltage [V]"](step.t))
            / (2 * h)
            for step, step_plus, step_minus in zip(
                steps(solution), steps(plus), steps(minus), strict=True
            )
        ]
        scale = max(np.max(np.abs(fd)) for fd in finite_differences)
        for step, finite_difference in zip(
            steps(solution), finite_differences, strict=True
        ):
            np.testing.assert_allclose(
                step["Voltage [V]"].sensitivities[name].ravel(),
                finite_difference,
                rtol=0,
                atol=2e-4 * scale,
            )
