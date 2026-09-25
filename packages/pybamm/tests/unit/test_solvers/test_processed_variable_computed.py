#
# Tests for the Processed Variable Computed class
#
# This class forms a container for variables (and sensitivities) calculated
#  by the idaklu solver, and does not possesses any capability to calculate
#  values itself since it does not have access to the full state vector
#

import typing
from functools import cache

import casadi
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pybamm
import tests
from pybamm.solvers.processed_variable import (
    ProcessedVariable0D,
    ProcessedVariable1D,
    ProcessedVariable2D,
    ProcessedVariable2DSciKitFEM,
    ProcessedVariable3D,
    ProcessedVariable3DSciKitFEM,
    ProcessedVariableRawFVM,
)


def to_casadi(var_pybamm, y, inputs=None):
    t_MX = casadi.MX.sym("t")
    y_MX = casadi.MX.sym("y", y.shape[0])

    inputs_MX_dict = {}
    inputs = inputs or {}
    for key, value in inputs.items():
        inputs_MX_dict[key] = casadi.MX.sym("input", value.shape[0])

    inputs_MX = casadi.vertcat(*[p for p in inputs_MX_dict.values()])

    var_sym = var_pybamm.to_casadi(t_MX, y_MX, inputs=inputs_MX_dict)

    var_casadi = casadi.Function("variable", [t_MX, y_MX, inputs_MX], [var_sym])
    return var_casadi


def process_and_check_2D_variable(
    var, first_spatial_var, second_spatial_var, disc=None, geometry_options=None
):
    # first_spatial_var should be on the "smaller" domain, i.e "r" for an "r-x" variable
    if geometry_options is None:
        geometry_options = {}
    if disc is None:
        disc = tests.get_discretisation_for_testing()
    disc.set_variable_slices([var])

    first_sol = disc.process_symbol(first_spatial_var).entries[:, 0]
    second_sol = disc.process_symbol(second_spatial_var).entries[:, 0]

    # Keep only the first iteration of entries
    first_sol = first_sol[: len(first_sol) // len(second_sol)]
    var_sol = disc.process_symbol(var)
    t_sol = np.linspace(0, 1)
    y_sol = np.ones(len(second_sol) * len(first_sol))[:, np.newaxis] * np.linspace(0, 5)

    var_casadi = to_casadi(var_sol, y_sol)
    model = tests.get_base_model_with_battery_geometry(**geometry_options)
    pybamm.ProcessedVariableComputed(
        [var_sol],
        [var_casadi],
        [y_sol],
        pybamm.Solution(t_sol, y_sol, model, {}),
    )
    # NB: ProcessedVariableComputed does not interpret y in the same way as
    #  ProcessedVariable; a better test of equivalence is to check that the
    #  results are the same between IDAKLUSolver with (and without)
    #  output_variables. This is implemented in the integration test:
    #    tests/integration/test_solvers/test_idaklu_solver.py
    #    ::test_output_variables
    return y_sol, first_sol, second_sol, t_sol


_T_INTERP = np.linspace(0, 600, 4)
# The full-state path evaluates (x - 1) + 1, which rounds values within about
# 1e-16 of zero to zero
_TOLERANCES = {"rtol": 1e-6, "atol": 1e-12}


def _dfn_with_time_integrals():
    model = pybamm.lithium_ion.DFN()
    voltage = model.variables["Voltage [V]"]
    # A nonzero initial condition must be counted once when segments are joined
    model.variables["Charge throughput [A.s]"] = pybamm.ExplicitTimeIntegral(
        model.variables["Current [A]"], pybamm.Scalar(100)
    )
    model.variables["Squared voltage integral [V2.s2]"] = (
        pybamm.ExplicitTimeIntegral(voltage, pybamm.Scalar(0)) ** 2
    )
    data = pybamm.DiscreteTimeData(
        _T_INTERP, np.full_like(_T_INTERP, 3.7), "Voltage data"
    )
    model.variables["Voltage sum of squares [V2]"] = pybamm.DiscreteTimeSum(
        (voltage - data) ** 2
    )
    return model


_OUTPUT_VARIABLE_MODELS = {
    "DFN": (
        _dfn_with_time_integrals,
        lambda: pybamm.ParameterValues("Marquis2019"),
        {"x_n": 4, "x_s": 3, "x_p": 4, "r_n": 5, "r_p": 5},
    ),
    "DFN size distribution": (
        lambda: pybamm.lithium_ion.DFN({"particle size": "distribution"}),
        lambda: pybamm.get_size_distribution_parameters(
            pybamm.ParameterValues("Marquis2019")
        ),
        {"x_n": 3, "x_s": 2, "x_p": 3, "r_n": 4, "r_p": 4, "R_n": 3, "R_p": 3},
    ),
    "SPMe 2+1D": (
        lambda: pybamm.lithium_ion.SPMe(
            {"current collector": "potential pair", "dimensionality": 2}
        ),
        lambda: pybamm.ParameterValues("Marquis2019"),
        # y and z differ, so swapping them changes the layout
        {"x_n": 3, "x_s": 2, "x_p": 3, "r_n": 4, "r_p": 4, "y": 3, "z": 4},
    ),
}

_OUTPUT_VARIABLE_CASES = [
    pytest.param("DFN", "Voltage [V]", ProcessedVariable0D, None, id="0D"),
    pytest.param(
        "DFN",
        "Electrolyte concentration [mol.m-3]",
        ProcessedVariable1D,
        None,
        id="1D-x",
    ),
    pytest.param(
        "DFN",
        "Negative particle concentration [mol.m-3]",
        ProcessedVariable2D,
        None,
        id="2D-r-x",
    ),
    # The zero-flux boundary edges are structural zeros, which the solver leaves out
    pytest.param(
        "DFN",
        "Electrolyte current density [A.m-2]",
        ProcessedVariable1D,
        None,
        id="1D-x-sparse",
    ),
    pytest.param(
        "DFN",
        "Negative particle flux [mol.m-2.s-1]",
        ProcessedVariable2D,
        None,
        id="2D-r-x-sparse",
    ),
    pytest.param(
        "SPMe 2+1D",
        "Negative current collector potential [V]",
        ProcessedVariable2DSciKitFEM,
        None,
        id="2D-y-z",
    ),
    pytest.param(
        "DFN size distribution",
        "Negative particle concentration distribution [mol.m-3]",
        ProcessedVariable3D,
        None,
        id="3D-r-R-x",
    ),
    pytest.param(
        "SPMe 2+1D",
        "Electrolyte concentration [mol.m-3]",
        ProcessedVariable3DSciKitFEM,
        None,
        id="3D-x-y-z",
    ),
    pytest.param(
        "DFN",
        "Charge throughput [A.s]",
        ProcessedVariable0D,
        "continuous",
        id="0D-time-integral",
    ),
    pytest.param(
        "DFN",
        "Squared voltage integral [V2.s2]",
        ProcessedVariable0D,
        "continuous",
        id="0D-time-integral-post-sum",
    ),
    pytest.param(
        "DFN",
        "Voltage sum of squares [V2]",
        ProcessedVariable0D,
        "discrete",
        id="0D-discrete-time-sum",
    ),
]

# ProcessedVariableComputed has no initialiser for these layouts
_NOT_COMPUTABLE = {
    pybamm.ProcessedVariable2DFVM,
    ProcessedVariableRawFVM,
    pybamm.ProcessedVariableUnstructured,
    pybamm.ProcessedVariableUnstructuredFVM,
}


# A DiscreteTimeSum, or a function of a time integral, has no value over joined segments
_NOT_JOINABLE = {"Squared voltage integral [V2.s2]", "Voltage sum of squares [V2]"}


@cache
def _solve(model_key, t_start=0.0, output_variables=()):
    """
    Solve a model over the 600 s from ``t_start``.

    Parameters
    ----------
    model_key : str
        Key into ``_OUTPUT_VARIABLE_MODELS``.
    t_start : float, optional
        Start time [s].
    output_variables : tuple of str, optional
        Variables for the solver to return. If empty, the solver returns the full
        state vector.

    Returns
    -------
    :class:`pybamm.Solution`
    """
    model, parameter_values, var_pts = _OUTPUT_VARIABLE_MODELS[model_key]
    sim = pybamm.Simulation(
        model(),
        parameter_values=parameter_values(),
        var_pts=var_pts,
        solver=pybamm.IDAKLUSolver(output_variables=list(output_variables)),
    )
    return sim.solve([t_start, t_start + 600], t_interp=t_start + _T_INTERP)


def _solve_full_and_computed(model_key):
    """
    Solve a model once with the full state vector and once with its cases as
    ``output_variables``.

    Parameters
    ----------
    model_key : str
        Key into ``_OUTPUT_VARIABLE_MODELS``.

    Returns
    -------
    tuple of :class:`pybamm.Solution`
        The full solution and the output_variables solution.
    """
    names = tuple(
        case.values[1] for case in _OUTPUT_VARIABLE_CASES if case.values[0] == model_key
    )
    return _solve(model_key), _solve(model_key, output_variables=names)


class TestProcessedVariableComputed:
    def test_processed_variable_0D(self):
        # without space
        y = pybamm.StateVector(slice(0, 1))
        var = y
        t_sol = np.array([0])
        y_sol = np.array([1])[:, np.newaxis]
        var_casadi = to_casadi(var, y_sol)
        sol = pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {})
        processed_var = pybamm.ProcessedVariableComputed(
            [var],
            [var_casadi],
            [y_sol],
            sol,
        )
        # Assert that the processed variable is the same as the solution
        np.testing.assert_array_equal(processed_var.entries, y_sol[0])
        # Check that 'data' produces the same output as 'entries'
        np.testing.assert_array_equal(processed_var.entries, processed_var.data)

        # Check unroll function
        np.testing.assert_array_equal(processed_var.unroll(), y_sol[0])

        # Check cumtrapz workflow produces no errors
        processed_var.cumtrapz_ic = 1
        processed_var.entries

        # check _update
        t_sol2 = np.array([1])
        y_sol2 = np.array([2])[:, np.newaxis]
        var_casadi = to_casadi(var, y_sol2)
        sol_2 = pybamm.Solution(t_sol2, y_sol2, pybamm.BaseModel(), {})
        processed_var2 = pybamm.ProcessedVariableComputed(
            [var],
            [var_casadi],
            [y_sol2],
            sol_2,
        )

        comb_sol = sol + sol_2
        comb_var = processed_var.update(processed_var2, comb_sol)
        np.testing.assert_array_equal(comb_var.entries, np.append(y_sol, y_sol2))

    def _build_0D_var(self, t_sol=None):
        # entries are 5 * t, so linear interpolation is exact
        if t_sol is None:
            t_sol = np.linspace(0, 1)
        y_sol = 5 * t_sol[np.newaxis, :]
        var = pybamm.t * pybamm.StateVector(slice(0, 1))
        return pybamm.ProcessedVariableComputed(
            [var],
            [to_casadi(var, y_sol)],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        )

    @pytest.mark.parametrize(
        ("t_query", "shape"),
        [
            (0.5, ()),
            (np.float64(0.5), ()),
            (np.array(0.5), ()),
            ([0.9, 0.3], (2,)),
            (np.array([0.9, 0.3, 0.6]), (3,)),
            (np.array([-0.5, 0.0, 0.5, 1.0, 2.0]), (5,)),
        ],
    )
    def test_0D_call_matches_the_xarray_route(self, t_query, shape):
        processed_var = self._build_0D_var()
        values = processed_var(t_query)
        expected = processed_var._xr_data_array.interp(t=t_query).values

        assert isinstance(values, np.ndarray)
        assert values.shape == shape
        np.testing.assert_allclose(values, expected, rtol=1e-14)
        t_array = np.asarray(t_query)
        in_range = (t_array >= 0) & (t_array <= 1)
        np.testing.assert_allclose(values[in_range], 5 * t_array[in_range])
        # out-of-range queries are NaN, as xarray fills them
        assert np.isnan(values[~in_range]).all()

    def test_0D_call_keeps_xarray_errors(self):
        # a multi-dimensional t is rejected, as for every other dimension
        processed_var = self._build_0D_var()
        with pytest.raises(IndexError, match=r"multi-dimensional"):
            processed_var(np.array([[0.1, 0.2], [0.3, 0.4]]))

        # a repeated solution time has no single value to interpolate from
        processed_var = self._build_0D_var(np.array([0.0, 0.5, 0.5, 1.0]))
        with pytest.raises(pd.errors.InvalidIndexError):
            processed_var(0.25)

    def test_data_array_is_built_on_first_interpolation(self):
        # .data, .entries and 0D time-only reads never build the xr.DataArray
        processed_var = self._build_0D_var()
        processed_var.data
        processed_var.entries
        processed_var(np.array([0.25, 0.75]))
        assert processed_var._xr_data_array_cache is None

        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [to_casadi(var_sol, y_sol)],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        )
        processed_var.entries
        assert processed_var._xr_data_array_cache is None

        processed_var(t_sol, x_sol)
        assert isinstance(processed_var._xr_data_array_cache, xr.DataArray)
        assert processed_var._xr_interp_args is None

    # check empty sensitivity works
    def test_processed_variable_0D_no_sensitivity(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariableComputed(
            [var],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        )

        # test no inputs (i.e. no sensitivity)
        assert processed_var.sensitivities == {}

        # with parameter
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        a = pybamm.InputParameter("a")
        var = t * y * a
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        inputs = {"a": np.array([1.0])}
        var_casadi = to_casadi(var, y_sol, inputs=inputs)
        processed_var = pybamm.ProcessedVariableComputed(
            [var],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), inputs),
        )

        # test no sensitivity raises error
        assert processed_var.sensitivities is None

    def test_processed_variable_1D(self):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        sol = pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {})
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [var_casadi],
            [y_sol],
            sol,
        )

        # Ordering from idaklu with output_variables set is different to
        # the full solver
        y_sol = y_sol.reshape((y_sol.shape[1], y_sol.shape[0])).transpose()
        np.testing.assert_array_equal(processed_var.entries, y_sol)
        np.testing.assert_array_equal(processed_var.entries, processed_var.data)
        np.testing.assert_allclose(
            processed_var(t_sol, x_sol), y_sol, rtol=1e-7, atol=1e-6
        )

        # Check unroll function
        np.testing.assert_array_equal(processed_var.unroll(), y_sol)

        # Check no error when data dimension is transposed vs node/edge
        processed_var.mesh.nodes, processed_var.mesh.edges = (
            processed_var.mesh.edges,
            processed_var.mesh.nodes,
        )
        processed_var.entries
        processed_var.mesh.nodes, processed_var.mesh.edges = (
            processed_var.mesh.edges,
            processed_var.mesh.nodes,
        )

        # Check that there are no errors with domain-specific attributes
        #  (see ProcessedVariableComputed.initialise_1D() for details)
        domain_list = [
            "particle",
            "separator",
            "current collector",
            "particle size",
            "random-non-specific-domain",
        ]
        for domain in domain_list:
            processed_var.domain = [domain]
            processed_var.entries

    def test_processed_variable_1D_unknown_domain(self):
        x = pybamm.SpatialVariable("x", domain="SEI layer", coord_sys="cartesian")
        geometry = pybamm.Geometry(
            {"SEI layer": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        )

        submesh_types = {"SEI layer": pybamm.Uniform1DSubMesh}
        var_pts = {x: 100}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        nt = 100

        y_sol = np.zeros((var_pts[x], nt))
        solution = pybamm.Solution(
            np.linspace(0, 1, nt),
            y_sol,
            pybamm.BaseModel(),
            {},
            np.linspace(0, 1, 1),
            np.zeros(var_pts[x]),
            "test",
        )

        c = pybamm.StateVector(slice(0, var_pts[x]), domain=["SEI layer"])
        c = c.with_mesh(mesh["SEI layer"])
        c_casadi = to_casadi(c, y_sol)
        pybamm.ProcessedVariableComputed([c], [c_casadi], [y_sol], solution)

    def test_processed_variable_1D_update(self):
        # variable 1
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol1 = disc.process_symbol(x).entries[:, 0]
        var_sol1 = disc.process_symbol(var)
        t_sol1 = np.linspace(0, 1)
        y_sol1 = np.ones_like(x_sol1)[:, np.newaxis] * np.linspace(0, 5)

        var_casadi1 = to_casadi(var_sol1, y_sol1)
        sol1 = pybamm.Solution(t_sol1, y_sol1, pybamm.BaseModel(), {})
        processed_var1 = pybamm.ProcessedVariableComputed(
            [var_sol1],
            [var_casadi1],
            [y_sol1],
            sol1,
        )

        # variable 2 -------------------
        var2 = pybamm.Variable("var2", domain=["negative electrode", "separator"])
        z = pybamm.SpatialVariable("z", domain=["negative electrode", "separator"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var2])
        z_sol2 = disc.process_symbol(z).entries[:, 0]
        var_sol2 = disc.process_symbol(var2)
        t_sol2 = np.linspace(2, 3)
        y_sol2 = np.ones_like(z_sol2)[:, np.newaxis] * np.linspace(5, 1)

        var_casadi2 = to_casadi(var_sol2, y_sol2)
        sol2 = pybamm.Solution(t_sol2, y_sol2, pybamm.BaseModel(), {})
        var_2 = pybamm.ProcessedVariableComputed(
            [var_sol2],
            [var_casadi2],
            [y_sol2],
            sol2,
        )

        comb_sol = sol1 + sol2
        comb_var = processed_var1.update(var_2, comb_sol)

        # Ordering from idaklu with output_variables set is different to
        # the full solver
        y_sol1 = y_sol1.reshape((y_sol1.shape[1], y_sol1.shape[0])).transpose()
        y_sol2 = y_sol2.reshape((y_sol2.shape[1], y_sol2.shape[0])).transpose()

        np.testing.assert_array_equal(
            comb_var.entries, np.concatenate((y_sol1, y_sol2), axis=1)
        )
        np.testing.assert_array_equal(comb_var.entries, comb_var.data)

    @pytest.mark.parametrize(
        ("t_later", "expected"),
        [([2, 3], [2.0, 2.0, 2.0, 2.0]), ([1, 2], [2.0, 2.0, 2.0])],
        ids=["gap", "shared-boundary"],
    )
    def test_processed_variable_0D_update_sensitivities(self, t_later, expected):
        def solve_processed_var(t_eval, calculate_sensitivities):
            model = pybamm.BaseModel()
            y = pybamm.Variable("y")
            a = pybamm.InputParameter("a")
            model.rhs = {y: 0 * y}
            model.initial_conditions = {y: 1}
            model.variables = {"a times y": a * y}

            solver = pybamm.IDAKLUSolver(output_variables=["a times y"])
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": 2.0},
                calculate_sensitivities=calculate_sensitivities,
                t_interp=np.array(t_eval),
            )
            return sol, sol["a times y"]

        _, processed_var_no_sens = solve_processed_var([0, 1], False)
        assert processed_var_no_sens.sensitivities == {}

        sol1, processed_var1 = solve_processed_var([0, 1], True)
        sol2, processed_var2 = solve_processed_var(t_later, True)

        combined_sol = sol1 + sol2
        combined_var = processed_var1.update(processed_var2, combined_sol)

        np.testing.assert_array_equal(combined_var.entries, np.array(expected))
        np.testing.assert_array_equal(
            combined_var.sensitivities["a"], np.ones(len(expected))
        )
        assert len(combined_var.entries) == len(combined_sol.t)

    def test_time_integral_update_sums_segments(self):
        model = pybamm.BaseModel()
        y = pybamm.Variable("y")
        a = pybamm.InputParameter("a")
        model.rhs = {y: 0 * y}
        model.initial_conditions = {y: 1}
        model.variables = {
            "Integral": pybamm.ExplicitTimeIntegral(a * y, pybamm.Scalar(1))
        }
        solver = pybamm.IDAKLUSolver(output_variables=["Integral"])
        first, later = (
            solver.solve(model, t_eval, inputs={"a": 2.0}, calculate_sensitivities=True)
            for t_eval in ([0, 1], [1, 3])
        )

        combined = (first + later)["Integral"]

        # 1 plus a * y = 2 integrated over [0, 3]
        np.testing.assert_allclose(combined.entries, [7.0])
        np.testing.assert_allclose(combined(), [7.0])
        np.testing.assert_allclose(combined.sensitivities["a"], [3.0])
        np.testing.assert_allclose(combined.sensitivities["all"], [[3.0]])

    @pytest.mark.parametrize("first_output_variables", [False, True])
    def test_time_integral_update_leaves_out_a_gap(self, first_output_variables):
        def solve(t_eval, output_variables):
            model = pybamm.BaseModel()
            y = pybamm.Variable("y")
            a = pybamm.InputParameter("a")
            model.rhs = {y: 0 * y}
            model.initial_conditions = {y: 1}
            model.variables = {
                "Integral": pybamm.ExplicitTimeIntegral(a * y, pybamm.Scalar(0))
            }
            solver = pybamm.IDAKLUSolver(output_variables=output_variables)
            return solver.solve(
                model, t_eval, inputs={"a": 2.0}, calculate_sensitivities=True
            )

        first = solve([0, 1], ["Integral"] if first_output_variables else None)
        later = solve([2, 3], ["Integral"])

        combined = (first + later)["Integral"]

        # Each segment integrates a * y = 2 over its own unit interval; [1, 2] is
        # not integrated
        np.testing.assert_allclose(combined.entries, [4.0])
        np.testing.assert_allclose(combined.sensitivities["a"], [2.0])

    def test_processed_variable_2D_x_r(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        process_and_check_2D_variable(var, r, x, disc=disc)

    def test_processed_variable_2D_R_x(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        R = pybamm.SpatialVariable(
            "R",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])

        disc = tests.get_size_distribution_disc_for_testing()
        process_and_check_2D_variable(
            var,
            R,
            x,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
        )

    def test_processed_variable_2D_R_z(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["current collector"]},
        )
        R = pybamm.SpatialVariable(
            "R",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["current collector"]},
        )
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_size_distribution_disc_for_testing()
        process_and_check_2D_variable(
            var,
            R,
            z,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
        )

    def test_processed_variable_2D_r_R(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative particle size"]},
        )
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative particle size"]},
        )
        R = pybamm.SpatialVariable("R", domain=["negative particle size"])

        disc = tests.get_size_distribution_disc_for_testing()
        process_and_check_2D_variable(
            var,
            r,
            R,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
        )

    def test_processed_variable_2D_x_z(self):
        var = pybamm.Variable(
            "var",
            domain=["negative electrode", "separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        x = pybamm.SpatialVariable(
            "x",
            domain=["negative electrode", "separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_1p1d_discretisation_for_testing()
        y_sol, x_sol, z_sol, t_sol = process_and_check_2D_variable(var, x, z, disc=disc)
        del x_sol

        # On edges
        x_s_edge = pybamm.Matrix(
            np.tile(disc.mesh["separator"].edges, len(z_sol)),
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        x_s_edge = x_s_edge.with_mesh(
            disc.mesh["separator"], secondary_mesh=disc.mesh["current collector"]
        )
        x_s_casadi = to_casadi(x_s_edge, y_sol)
        processed_x_s_edge = pybamm.process_variable(
            "test",
            [x_s_edge],
            [x_s_casadi],
            pybamm.Solution(
                t_sol, y_sol, tests.get_base_model_with_battery_geometry(), {}
            ),
        )
        np.testing.assert_array_equal(
            x_s_edge.entries.flatten(), processed_x_s_edge.entries[:, :, 0].T.flatten()
        )

    def test_processed_variable_2D_space_only(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.array([0])
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis]

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            processed_var.data,
        )

        # Check unroll function (2D)
        np.testing.assert_array_equal(processed_var.unroll(), y_sol.reshape(10, 40, 1))

    def test_processed_variable_2D_fixed_t_scikit(self):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y = disc.mesh["current collector"].edges["y"]
        z = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol = var_sol.with_mesh(disc.mesh["current collector"])
        t_sol = np.array([0])
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [var_casadi],
            [u_sol],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
        )

    def test_processed_variable_3D_r_R_x(self):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode"],
            },
        )
        disc = tests.get_size_distribution_disc_for_testing(xpts=3, rpts=4, Rpts=5)
        disc.set_variable_slices([var])
        x_sol = disc.mesh["negative electrode"].nodes
        R_sol = disc.mesh["negative particle size"].nodes
        r_sol = disc.mesh["negative particle"].nodes
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1, 2)
        u_sol = np.ones(len(x_sol) * len(R_sol) * len(r_sol))[:, np.newaxis] * t_sol

        var_casadi = to_casadi(var_sol, u_sol)
        geometry_options = {"options": {"particle size": "distribution"}}
        model = tests.get_base_model_with_battery_geometry(**geometry_options)
        # base_variables_data is time-major (n_t, output); u_sol is (output, n_t)
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [var_casadi],
            [u_sol.T],
            pybamm.Solution(t_sol, u_sol, model, {}),
        )

        # Check shape (prim, sec, ter, time)
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(u_sol, [len(r_sol), len(R_sol), len(x_sol), len(t_sol)]),
        )

        # Check unroll function (3D)
        np.testing.assert_array_equal(processed_var.unroll(), u_sol.reshape(4, 5, 3, 2))

    @pytest.mark.parametrize("edges_eval", [False, True])
    def test_processed_variable_3D_x_y_z(self, edges_eval):
        disc = tests.get_2p1d_discretisation_for_testing(xpts=5, ypts=6, zpts=7)
        if edges_eval:
            var_cc = pybamm.Variable("var_cc", domain=["current collector"])
            var = pybamm.PrimaryBroadcastToEdges(var_cc, ["negative electrode"])
            x_sol = disc.mesh["negative electrode"].edges
            disc.set_variable_slices([var_cc])
        else:
            var = pybamm.Variable(
                "var",
                domain=["negative electrode"],
                auxiliary_domains={"secondary": ["current collector"]},
            )
            x_sol = disc.mesh["negative electrode"].nodes
            disc.set_variable_slices([var])

        Nx = len(x_sol)
        y_sol = disc.mesh["current collector"].edges["y"]
        z_sol = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1, 2)
        u_sol = np.ones(len(x_sol) * len(y_sol) * len(z_sol))[:, np.newaxis] * t_sol

        var_casadi = to_casadi(var_sol, u_sol)
        # base_variables_data is time-major (n_t, output); u_sol is (output, n_t)
        processed_var = pybamm.ProcessedVariableComputed(
            [var_sol],
            [var_casadi],
            [u_sol.T],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
        )

        # Check shape (prim, sec, ter, time)
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(u_sol, [len(x_sol), len(y_sol), len(z_sol), len(t_sol)]),
        )

        # Check unroll function (3D)
        np.testing.assert_array_equal(
            processed_var.unroll(), u_sol.reshape(Nx, 6, 7, 2)
        )

    @pytest.mark.parametrize(
        ("model_key", "name", "layout", "time_integral_method"), _OUTPUT_VARIABLE_CASES
    )
    def test_output_variable_matches_full_solve(
        self, model_key, name, layout, time_integral_method
    ):
        full_solution, computed_solution = _solve_full_and_computed(model_key)
        full, computed = full_solution[name], computed_solution[name]

        # A routing change must not quietly move a case off the path it covers
        assert type(full) is layout
        method = full.time_integral.method if full.time_integral else None
        assert method == time_integral_method

        assert isinstance(computed, pybamm.ProcessedVariableComputed)
        np.testing.assert_allclose(computed.entries, full.entries, **_TOLERANCES)

        # Solution.__add__ merges with an output_variables solve via as_computed()
        converted = full.as_computed()
        assert converted.time_indep == computed.time_indep
        np.testing.assert_array_equal(converted.entries, full.entries)
        t = None if computed.time_indep else np.array([150.0, 450.0])
        np.testing.assert_allclose(converted(t=t), computed(t=t), **_TOLERANCES)

    @pytest.mark.parametrize("first_output_variables", [False, True])
    @pytest.mark.parametrize(
        ("model_key", "name", "layout", "time_integral_method"), _OUTPUT_VARIABLE_CASES
    )
    def test_output_variable_joins_at_shared_boundary(
        self, model_key, name, layout, time_integral_method, first_output_variables
    ):
        full_solution = _solve(model_key)
        # Both later segments start at 600 s, where the first ends
        later_full = _solve(model_key, t_start=600.0)
        later_computed = _solve(model_key, t_start=600.0, output_variables=(name,))
        if first_output_variables:
            first = _solve(model_key, output_variables=(name,))
        else:
            # copy() drops full-state variables other tests read from the cached solve
            first = full_solution.copy()

        joined = (first + later_computed)[name]
        if name in _NOT_JOINABLE:
            with pytest.raises(NotImplementedError, match=r"ExplicitTimeIntegral"):
                joined.entries
            return

        # A full-state sum evaluates the variable on the joined states, not via _concat
        expected = (full_solution + later_full)[name]
        np.testing.assert_allclose(joined.entries, expected.entries, **_TOLERANCES)
        t = None if joined.time_indep else np.array([150.0, 750.0])
        np.testing.assert_allclose(
            joined(t=t), expected.as_computed()(t=t), **_TOLERANCES
        )

    def test_output_variable_cases_cover_every_layout(self):
        subclasses, stack = set(), [pybamm.ProcessedVariable]
        while stack:
            for subclass in stack.pop().__subclasses__():
                subclasses.add(subclass)
                stack.append(subclass)
        layouts = {c for c in subclasses if c.__module__.startswith("pybamm")}
        covered = {case.values[2] for case in _OUTPUT_VARIABLE_CASES}
        missing = layouts - covered - _NOT_COMPUTABLE
        assert not missing, (
            "Add a case to _OUTPUT_VARIABLE_CASES, or add the class to "
            "_NOT_COMPUTABLE: " + ", ".join(sorted(c.__name__ for c in missing))
        )

        methods = typing.get_args(
            typing.get_type_hints(pybamm.ProcessedVariableTimeIntegral)["method"]
        )
        covered_methods = {case.values[3] for case in _OUTPUT_VARIABLE_CASES}
        assert set(methods) <= covered_methods, (
            "Add a time-integral case to _OUTPUT_VARIABLE_CASES for: "
            + ", ".join(sorted(set(methods) - covered_methods))
        )
