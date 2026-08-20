"""Layout-contract tests for native observation (pure Python, no Rust producer)."""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import OutputAssembly


def _solve_spm():
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)
    return sim.solve([0, 3600])


class TestUnrollNnzDense:
    def test_unroll_nnz_returns_dense_when_no_casadi(self):
        # A ProcessedVariableComputed built with base_variables_casadi=[None], the
        # native contract, must not inspect CasADi sparsity.
        sol = _solve_spm()
        var = pybamm.StateVector(slice(0, 1))
        n_t = len(sol.t)
        data = [np.arange(n_t, dtype=float).reshape(n_t, 1)]
        pvc = pybamm.ProcessedVariableComputed([var], [None], data, sol)

        # Directly test _unroll_nnz - it should return data as-is when no CasADi func
        result = pvc._unroll_nnz(data)
        np.testing.assert_array_equal(result[0], data[0])


class TestLayoutContract:
    def test_0d_time_major(self):
        # Contract: base_variables_data is one (n_times, output_size) array per
        # sub-solution; for 0D output_size == 1 and data[t, 0] is the value at t.
        sol = _solve_spm()
        base = [pybamm.StateVector(slice(0, 1))]
        n_t = len(sol.t)
        values = np.linspace(3.0, 4.2, n_t)
        data = [values.reshape(n_t, 1)]  # time-major, C-contiguous
        pvc = pybamm.ProcessedVariableComputed(base, [None], data, sol)
        np.testing.assert_allclose(pvc.entries.reshape(-1), values)

    def test_1d_time_major_unrolls_to_space_by_time(self):
        # A 1D variable's (n_times, len_space) array must unroll to
        # (len_space, n_times) via reshape((n_times, len_space)).transpose().
        sol = _solve_spm()
        base_pv = sol["X-averaged negative particle concentration [mol.m-3]"]
        var = base_pv.base_variables[0]
        len_space = var.shape[0]  # 20 radial nodes in the negative particle
        n_t = len(sol.t)
        rng = np.arange(n_t * len_space, dtype=float).reshape(n_t, len_space)
        pvc = pybamm.ProcessedVariableComputed([var], [None], [rng], sol)
        # entries is (len_space, n_t): element [k, t] == rng[t, k]
        assert pvc.entries.shape[0] == len_space
        np.testing.assert_allclose(pvc.entries[:, 0], rng[0, :])
        np.testing.assert_allclose(pvc.entries[:, -1], rng[-1, :])


class TestOutputAssembly:
    """The outputs-only payload layout, shared by every solver that produces one.

    A scalar, a 20-component vector and a second scalar: an ordinal-indexed
    reader returns one component for the vector and shifts the scalar after it.
    """

    _NAMES = [
        "Voltage [V]",
        "X-averaged negative particle concentration [mol.m-3]",
        "Current [A]",
    ]

    @staticmethod
    def _fixture(names=None):
        """An SPM solution, an assembly over ``names``, and a row-indexed payload."""
        names = names or TestOutputAssembly._NAMES
        solution = _solve_spm()
        model = solution.all_models[0]
        lens = [
            int(np.prod(model.get_processed_variable_or_event(name).shape))
            for name in names
        ]
        assembly = OutputAssembly(names, lens)
        # data[t, k] == k, so a variable's entries are its own row indices.
        data = np.tile(np.arange(assembly.n_rows, dtype=float), (len(solution.t), 1))
        return assembly, solution, data

    def test_rows_are_sliced_by_component_count_not_ordinal(self):
        assembly, solution, data = self._fixture()
        assembly.attach(solution, data)

        np.testing.assert_allclose(solution["Voltage [V]"].entries, 0.0)
        np.testing.assert_allclose(
            solution[self._NAMES[1]].entries[:, 0], np.arange(1.0, 21.0)
        )
        # 21, not 2: the vector consumed rows 1--20 rather than row 1 alone.
        np.testing.assert_allclose(solution["Current [A]"].entries, 21.0)

    def test_a_payload_of_the_wrong_width_is_rejected(self):
        assembly, solution, data = self._fixture()
        with pytest.raises(pybamm.SolverError, match=r"Output row count mismatch"):
            assembly.attach(solution, data[:, :-1])

    def test_sensitivities_are_named_and_flattened_per_parameter(self):
        assembly, solution, data = self._fixture(["Voltage [V]"])
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
        assert attached["all"].shape == (n_t, 2)
        np.testing.assert_allclose(attached["a"], sensitivities[:, 0, 0])
        np.testing.assert_allclose(attached["b"], sensitivities[:, 0, 1])

    def test_sensitivities_of_the_wrong_shape_are_rejected(self):
        assembly, solution, data = self._fixture(["Voltage [V]"])
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

    def test_one_block_per_parameter_is_required(self):
        assembly, solution, _ = self._fixture(["Voltage [V]"])
        n_t = len(solution.t)
        with pytest.raises(
            pybamm.SolverError, match=r"Sensitivity block count mismatch"
        ):
            assembly.stack_parameter_blocks([np.zeros(n_t)], n_t, ["a", "b"])

    def test_parameter_blocks_stack_into_the_attachable_layout(self):
        assembly, solution, _ = self._fixture(["Voltage [V]"])
        n_t = len(solution.t)
        blocks = [np.arange(n_t, dtype=float), np.arange(n_t, dtype=float) * -1.0]

        stacked = assembly.stack_parameter_blocks(blocks, n_t, ["a", "b"])

        assert stacked.shape == (n_t, 1, 2)
        np.testing.assert_allclose(stacked[:, 0, 0], blocks[0])
        np.testing.assert_allclose(stacked[:, 0, 1], blocks[1])

    def test_a_solve_without_sensitivities_leaves_an_empty_mapping(self):
        # Not None: an outputs-only solve retains no state to compute them from,
        # so the answer is "there are none", not "ask again later".
        assembly, solution, data = self._fixture(["Voltage [V]"])
        assembly.attach(solution, data)
        assert solution["Voltage [V]"]._sensitivities == {}
