"""Tests for the row layout of ``output_variables`` payloads."""

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import OutputAssembly
from pybamm.solvers.variable_observer import SampledObserver


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
        # One (n_times, output_size) array per sub-solution
        base = [pybamm.StateVector(slice(0, 1))]
        n_t = len(solution.t)
        values = np.linspace(3.0, 4.2, n_t)
        data = [values.reshape(n_t, 1)]
        pvc = pybamm.ProcessedVariableComputed(
            base, [_dense_function(1)], data, solution
        )
        np.testing.assert_allclose(pvc.entries.reshape(-1), values)

    def test_1d_time_major_unrolls_to_space_by_time(self, solution):
        # A 1D variable's (n_times, len_space) rows unroll to (len_space, n_times)
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

        # The field: the property short-circuits to {} on this input-free solve
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
        # Empty, not None: an outputs-only solve keeps no state to compute them from
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


_DOMAIN = "negative electrode"
# Both solves take the same steps, so only roundoff separates their values
_TOLERANCES = {"rtol": 1e-10, "atol": 1e-12}


def _finite_volume_model(dimension):
    """A decaying field that varies across the cells, and variables of it."""
    x = pybamm.SpatialVariable(
        "x_n", domain=[_DOMAIN], coord_sys="cartesian", direction="lr"
    )
    z = pybamm.SpatialVariable(
        "z_n", domain=[_DOMAIN], coord_sys="cartesian", direction="tb"
    )
    spatial_variables = [x, z]
    if dimension == 3:
        y = pybamm.SpatialVariable("y_n", domain=[_DOMAIN], coord_sys="cartesian")
        spatial_variables = [x, y, z]
    u = pybamm.Variable("u", domain=[_DOMAIN])
    model = pybamm.BaseModel()
    model.rhs = {u: -pybamm.InputParameter("k") * (1 + x + 2 * z) * u}
    model.initial_conditions = {u: 1 + x * z}
    model.variables = {
        "u": u,
        "grad u": pybamm.grad(u),
        "Time [s]": pybamm.t,
        "Time integral of u": pybamm.ExplicitTimeIntegral(u, pybamm.Scalar(0)),
    }
    # Unequal point counts, so a transposed layout would not line up
    geometry = {
        _DOMAIN: {
            var: {"min": 0, "max": 1 + k} for k, var in enumerate(spatial_variables)
        }
    }
    var_pts = dict(
        zip(spatial_variables, [3, 4, 5][: len(spatial_variables)], strict=True)
    )
    return model, geometry, var_pts


_FINITE_VOLUME_MESHES = {
    "unstructured 2D": (
        2,
        pybamm.UnstructuredMeshGenerator,
        pybamm.FiniteVolumeUnstructured,
    ),
    "unstructured 3D": (
        3,
        pybamm.UnstructuredMeshGenerator,
        pybamm.FiniteVolumeUnstructured,
    ),
    "structured 2D": (2, lambda: pybamm.Uniform2DSubMesh, pybamm.FiniteVolume2D),
}


def _solve_on_mesh(mesh_key, output_variables=None, calculate_sensitivities=False):
    dimension, submesh_type, spatial_method = _FINITE_VOLUME_MESHES[mesh_key]
    model, geometry, var_pts = _finite_volume_model(dimension)
    sim = pybamm.Simulation(
        model,
        geometry=geometry,
        submesh_types={_DOMAIN: submesh_type()},
        var_pts=var_pts,
        spatial_methods={_DOMAIN: spatial_method()},
        # output_variables is an IDAKLUSolver option
        solver=pybamm.IDAKLUSolver(output_variables=output_variables),
    )
    return sim.solve(
        [0, 1], inputs={"k": 1.0}, calculate_sensitivities=calculate_sensitivities
    )


class TestOutputsOnFiniteVolume2DMeshes:
    """Outputs whose mesh has no 1D node grid read like a full-state solve."""

    @pytest.mark.parametrize("mesh_key", list(_FINITE_VOLUME_MESHES))
    def test_scalar_output_matches_full_solve(self, mesh_key):
        full = _solve_on_mesh(mesh_key)["u"]
        output = _solve_on_mesh(mesh_key, output_variables=["u"])["u"]

        assert type(output) is type(full)
        np.testing.assert_allclose(output.entries, full.entries, **_TOLERANCES)
        # At the solve's own times; off them the full solve interpolates in states
        t = full.t_pts[[2, -1]]
        np.testing.assert_allclose(
            output(t=t, x=[0.2, 0.7], z=[0.3, 1.5]),
            full(t=t, x=[0.2, 0.7], z=[0.3, 1.5]),
            **_TOLERANCES,
        )

    @pytest.mark.parametrize("mesh_key", ["unstructured 2D", "unstructured 3D"])
    def test_vector_field_output_matches_full_solve(self, mesh_key):
        full = _solve_on_mesh(mesh_key)["grad u"]
        output = _solve_on_mesh(mesh_key, output_variables=["grad u"])["grad u"]

        assert isinstance(output, pybamm.ProcessedVariableVectorFieldUnstructuredFVM)
        assert output.n_components == full.n_components
        for output_component, full_component in zip(
            output.entries, full.entries, strict=True
        ):
            np.testing.assert_allclose(output_component, full_component, **_TOLERANCES)

    @pytest.mark.parametrize("mesh_key", ["unstructured 2D", "structured 2D"])
    def test_scalar_output_sensitivities_match_full_solve(self, mesh_key):
        full = _solve_on_mesh(mesh_key, calculate_sensitivities=True)["u"]
        output = _solve_on_mesh(
            mesh_key, output_variables=["u"], calculate_sensitivities=True
        )["u"]

        np.testing.assert_allclose(
            output.sensitivities["k"], full.sensitivities["k"], **_TOLERANCES
        )

    def test_vector_field_output_sensitivities_match_full_solve(self):
        full = _solve_on_mesh("unstructured 2D", calculate_sensitivities=True)
        output = _solve_on_mesh(
            "unstructured 2D",
            output_variables=["grad u"],
            calculate_sensitivities=True,
        )

        for output_component, full_component in zip(
            output["grad u"]._component_vars,
            full["grad u"]._component_vars,
            strict=True,
        ):
            np.testing.assert_allclose(
                output_component.sensitivities["k"],
                full_component.sensitivities["k"],
                **_TOLERANCES,
            )

    @pytest.mark.parametrize("mesh_key", ["unstructured 2D", "structured 2D"])
    def test_time_integral_output_is_not_supported(self, mesh_key):
        # A full-state read of it on a structured mesh returns the integrand
        with pytest.raises(NotImplementedError, match=r"time integrals"):
            _solve_on_mesh(mesh_key, output_variables=["Time integral of u"])

    def test_copy_keeps_every_output(self):
        solution = _solve_on_mesh("unstructured 2D", output_variables=["u", "Time [s]"])
        expected = {name: solution[name].entries for name in ["u", "Time [s]"]}

        copied = solution.copy()

        assert isinstance(copied["Time [s]"], pybamm.ProcessedVariableComputed)
        for name, entries in expected.items():
            np.testing.assert_array_equal(copied[name].entries, entries)

    def test_vector_field_output_on_structured_mesh_is_not_supported(self):
        with pytest.raises(NotImplementedError, match=r"structured 2D finite-volume"):
            _solve_on_mesh("structured 2D", output_variables=["grad u"])


class TestSampledObserver:
    def test_component_owns_its_rows_and_leaf(self):
        t, y, p = (casadi.MX.sym(name) for name in "typ")
        leaf = casadi.Function(
            "vector", [t, y, p], [casadi.vertcat(y, 2 * y, 3 * y, 4 * y)]
        )
        samples = np.arange(12, dtype=float).reshape(3, 4)

        component = SampledObserver([leaf], samples).component(1, 2)

        np.testing.assert_array_equal(component._samples, samples[:, 2:])
        (component_leaf,) = component.leaves
        np.testing.assert_allclose(component_leaf(0, 5, 0).full().ravel(), [15, 20])
