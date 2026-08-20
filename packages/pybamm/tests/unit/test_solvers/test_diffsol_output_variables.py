"""Diffsol ``output_variables``: layout parity against the full-state solve.

Rust returns one row per flattened output component, so a vector output variable
occupies several rows. These tests pin that the Python side slices by component
count rather than by variable ordinal, and that the results expose the same
interpolating interface as every other solver.
"""

import numpy as np
import pytest

import pybamm

# A 20-node spatial variable sandwiched between two scalars: an ordinal-indexed
# reader returns one component for it and shifts everything after it.
_VECTOR_VAR = "Negative particle surface concentration [mol.m-3]"
_OUTPUT_VARIABLES = ["Voltage [V]", _VECTOR_VAR, "Current [A]"]

_SOLVER_TOL = 1e-8


class TestDiffsolOutputVariables:
    @pytest.fixture(scope="class")
    def solutions(self):
        """Solve SPM twice: once output-only, once for the full state."""
        t_eval = np.linspace(0, 600, 11)

        model_outputs = pybamm.lithium_ion.SPM()
        sol_outputs = pybamm.Simulation(
            model_outputs,
            solver=pybamm.DiffsolSolver(
                rtol=_SOLVER_TOL,
                atol=_SOLVER_TOL,
                output_variables=_OUTPUT_VARIABLES,
            ),
        ).solve(t_eval)

        model_full = pybamm.lithium_ion.SPM()
        sol_full = pybamm.Simulation(
            model_full,
            solver=pybamm.DiffsolSolver(rtol=_SOLVER_TOL, atol=_SOLVER_TOL),
        ).solve(t_eval)

        return sol_outputs, sol_full, t_eval

    def test_vector_output_keeps_every_component(self, solutions):
        sol_outputs, sol_full, _ = solutions

        expected = np.asarray(sol_full[_VECTOR_VAR].entries)
        actual = np.asarray(sol_outputs[_VECTOR_VAR].entries)

        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_scalar_after_a_vector_is_not_shifted(self, solutions):
        sol_outputs, sol_full, _ = solutions

        np.testing.assert_allclose(
            np.asarray(sol_outputs["Current [A]"].entries),
            np.asarray(sol_full["Current [A]"].entries),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_scalar_before_a_vector_is_unaffected(self, solutions):
        sol_outputs, sol_full, _ = solutions

        np.testing.assert_allclose(
            np.asarray(sol_outputs["Voltage [V]"].entries),
            np.asarray(sol_full["Voltage [V]"].entries),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_vector_output_primal_and_sensitivity_layouts_agree(self):
        """The sensitivity path always sliced by component count; the primal path
        sliced by variable ordinal. For a vector variable the two disagreed."""
        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")

        model = pybamm.lithium_ion.SPM()
        model.events = []
        sol = pybamm.Simulation(
            model,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(
                rtol=_SOLVER_TOL, atol=_SOLVER_TOL, output_variables=[_VECTOR_VAR]
            ),
        ).solve(
            np.linspace(0, 100, 6), inputs={"I": 0.5}, calculate_sensitivities=["I"]
        )

        variable = sol[_VECTOR_VAR]
        assert variable.entries.size == variable.sensitivities["I"].shape[0]

    def test_output_rows_stay_aligned_across_the_batch_window(self):
        """200 output points cross the 128-lane staging window in Rust; a flush
        off-by-one would shift every row after the window boundary."""
        t_eval = np.linspace(0, 600, 200)

        sol_outputs = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            solver=pybamm.DiffsolSolver(
                rtol=_SOLVER_TOL,
                atol=_SOLVER_TOL,
                output_variables=_OUTPUT_VARIABLES,
            ),
        ).solve(t_eval)
        sol_full = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            solver=pybamm.DiffsolSolver(rtol=_SOLVER_TOL, atol=_SOLVER_TOL),
        ).solve(t_eval)

        for name in _OUTPUT_VARIABLES:
            np.testing.assert_allclose(
                np.asarray(sol_outputs[name](t_eval)),
                np.asarray(sol_full[name](t_eval)),
                rtol=1e-6,
                atol=1e-8,
                err_msg=name,
            )

    @pytest.mark.parametrize("name", _OUTPUT_VARIABLES)
    def test_output_variables_are_callable(self, solutions, name):
        """`solution[name](t)` is the documented read interface for every solver.

        Compared against the full-state solve rather than against ``.entries``:
        for a spatial variable the call interface interpolates onto the node grid
        including its two boundary points, so the two shapes differ by design.
        """
        sol_outputs, sol_full, t_eval = solutions

        np.testing.assert_allclose(
            np.asarray(sol_outputs[name](t_eval)),
            np.asarray(sol_full[name](t_eval)),
            rtol=1e-6,
            atol=1e-8,
        )


class TestDiffsolOutputVariablesAcrossExperimentSteps:
    """Stitching outputs-only segments together, as an ``Experiment`` does."""

    @staticmethod
    def _solve(solver_cls, convert_to_format, **solver_kwargs):
        experiment = pybamm.Experiment(
            ["Discharge at 1C for 200 seconds", "Rest for 100 seconds"]
        )
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = convert_to_format
        return pybamm.Simulation(
            model,
            parameter_values=pybamm.ParameterValues("Chen2020"),
            experiment=experiment,
            solver=solver_cls(rtol=_SOLVER_TOL, atol=_SOLVER_TOL, **solver_kwargs),
        ).solve()

    def test_experiment_with_output_variables_stitches(self):
        # Segment stitching slices all_ys for every sub-solution; a None there
        # raised TypeError before diffsol supplied a zero-row array.
        sol = self._solve(
            pybamm.DiffsolSolver, "rust", output_variables=["Voltage [V]"]
        )
        assert len(sol.all_ts) == 2
        assert sol.variables_returned
        assert all(y.shape[0] == 0 for y in sol.all_ys)
        assert np.all(np.isfinite(np.asarray(sol["Voltage [V]"](sol.t))))

    def test_experiment_outputs_match_the_full_state_solve(self):
        sol_out = self._solve(
            pybamm.DiffsolSolver, "rust", output_variables=["Voltage [V]"]
        )
        sol_full = self._solve(pybamm.DiffsolSolver, "rust")
        t = sol_full.t
        np.testing.assert_allclose(
            np.asarray(sol_out["Voltage [V]"](t)),
            np.asarray(sol_full["Voltage [V]"](t)),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_restarting_with_sensitivities_is_refused_not_seeded_with_zeros(self):
        # An outputs-only step boundary carries no state sensitivities, so the
        # next step would silently restart dy0/dp from zero.
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "rust"
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values["Negative particle diffusivity [m2.s-1]"] = (
            pybamm.InputParameter("D_n")
        )
        solver = pybamm.DiffsolSolver(output_variables=["Voltage [V]"])
        simulation = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=solver
        )
        simulation.build()
        inputs = {"D_n": 3.3e-14}

        first = solver.step(
            pybamm.EmptySolution(),
            simulation.built_model,
            dt=100.0,
            inputs=inputs,
            calculate_sensitivities=["D_n"],
        )
        with pytest.raises(pybamm.SolverError, match=r"output variables only"):
            solver.step(
                first,
                simulation.built_model,
                dt=100.0,
                inputs=inputs,
                calculate_sensitivities=["D_n"],
            )
