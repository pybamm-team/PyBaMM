"""Tests for the row layout of ``output_variables`` payloads."""

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import OutputAssembly


@pytest.fixture(scope="module")
def spm_solution():
    return pybamm.Simulation(pybamm.lithium_ion.SPM()).solve([0, 3600])


@pytest.fixture
def solution(spm_solution):
    # A copy, so variables attached by one test do not leak into the next.
    return spm_solution.copy()


def _dense_function(size):
    """A CasADi ``f(t, y, p)`` with ``size`` dense rows."""
    t, y, p = (casadi.MX.sym(name) for name in "typ")
    return casadi.Function("dense", [t, y, p], [casadi.repmat(t + y + p, size, 1)])


class TestLayoutContract:
    def test_0d_time_major(self, solution):
        # One (n_times, output_size) array per sub-solution; for 0D
        # output_size == 1 and data[t, 0] is the value at t.
        base = [pybamm.StateVector(slice(0, 1))]
        n_t = len(solution.t)
        values = np.linspace(3.0, 4.2, n_t)
        data = [values.reshape(n_t, 1)]
        pvc = pybamm.ProcessedVariableComputed(
            base, [_dense_function(1)], data, solution
        )
        np.testing.assert_allclose(pvc.entries.reshape(-1), values)

    def test_1d_time_major_unrolls_to_space_by_time(self, solution):
        # A 1D variable's (n_times, len_space) array unrolls to
        # (len_space, n_times) via reshape((n_times, len_space)).transpose().
        base_pv = solution["X-averaged negative particle concentration [mol.m-3]"]
        var = base_pv.base_variables[0]
        len_space = var.shape[0]
        n_t = len(solution.t)
        rows = np.arange(n_t * len_space, dtype=float).reshape(n_t, len_space)
        pvc = pybamm.ProcessedVariableComputed(
            [var], [_dense_function(len_space)], [rows], solution
        )
        assert pvc.entries.shape[0] == len_space
        np.testing.assert_allclose(pvc.entries[:, 0], rows[0, :])
        np.testing.assert_allclose(pvc.entries[:, -1], rows[-1, :])


class TestOutputAssembly:
    """A scalar, a 20-component vector and a second scalar.

    An ordinal-indexed reader would return one component for the vector and
    shift the scalar after it.
    """

    _NAMES = [
        "Voltage [V]",
        "X-averaged negative particle concentration [mol.m-3]",
        "Current [A]",
    ]

    @staticmethod
    def _assembly(solution, names=None):
        """An assembly over ``names`` and a payload whose entries are row indices."""
        names = names or TestOutputAssembly._NAMES
        model = solution.all_models[0]
        casadi_fns = {
            name: _dense_function(
                int(np.prod(model.get_processed_variable_or_event(name).shape))
            )
            for name in names
        }
        assembly = OutputAssembly(names, casadi_fns)
        data = np.tile(np.arange(assembly.n_rows, dtype=float), (len(solution.t), 1))
        return assembly, data

    def test_rows_are_sliced_by_component_count_not_ordinal(self, solution):
        assembly, data = self._assembly(solution)
        assembly.attach(solution, data)

        np.testing.assert_allclose(solution["Voltage [V]"].entries, 0.0)
        np.testing.assert_allclose(
            solution[self._NAMES[1]].entries[:, 0], np.arange(1.0, 21.0)
        )
        # 21, not 2: the vector consumed rows 1--20 rather than row 1 alone.
        np.testing.assert_allclose(solution["Current [A]"].entries, 21.0)

    def test_a_payload_of_the_wrong_width_is_rejected(self, solution):
        assembly, data = self._assembly(solution)
        with pytest.raises(pybamm.SolverError, match=r"Output row count mismatch"):
            assembly.attach(solution, data[:, :-1])

    def test_sensitivities_are_named_and_flattened_per_parameter(self, solution):
        assembly, data = self._assembly(solution, ["Voltage [V]"])
        n_t = len(solution.t)
        sensitivities = np.arange(n_t * 2, dtype=float).reshape(n_t, 1, 2)

        assembly.attach(
            solution,
            data,
            sensitivities=sensitivities,
            sensitivity_names=["a", "b"],
        )

        # Read the field, not the property: the property short-circuits to {} on
        # this input-free SPM solve.
        attached = solution["Voltage [V]"]._sensitivities
        assert set(attached) == {"all", "a", "b"}
        assert attached["all"].shape == (n_t, 2)
        np.testing.assert_allclose(attached["a"], sensitivities[:, 0, 0])
        np.testing.assert_allclose(attached["b"], sensitivities[:, 0, 1])

    def test_sensitivities_of_the_wrong_shape_are_rejected(self, solution):
        assembly, data = self._assembly(solution, ["Voltage [V]"])
        n_t = len(solution.t)
        with pytest.raises(
            pybamm.SolverError, match=r"Output sensitivity shape mismatch"
        ):
            assembly.attach(
                solution,
                data,
                sensitivities=np.zeros((n_t, 1, 1)),
                sensitivity_names=["a", "b"],
            )

    def test_a_solve_without_sensitivities_leaves_an_empty_mapping(self, solution):
        # Not None: an outputs-only solve retains no state to compute them from,
        # so the answer is "there are none", not "ask again later".
        assembly, data = self._assembly(solution, ["Voltage [V]"])
        assembly.attach(solution, data)
        assert solution["Voltage [V]"]._sensitivities == {}

    def test_a_sparse_variable_owns_only_its_nonzero_rows(self, solution):
        t, y, p = (casadi.MX.sym(name) for name in "typ")
        sparse = casadi.Function(
            "sparse", [t, y, p], [casadi.vertcat(t, casadi.MX(1, 1))]
        )
        assembly = OutputAssembly(
            ["Voltage [V]", "Current [A]"],
            {"Voltage [V]": sparse, "Current [A]": _dense_function(1)},
        )
        assert assembly.n_rows == 2

        n_t = len(solution.t)
        with pytest.raises(pybamm.SolverError, match=r"sparse variable"):
            assembly.attach(
                solution,
                np.zeros((n_t, 2)),
                sensitivities=np.zeros((n_t, 2, 1)),
                sensitivity_names=["a"],
            )
