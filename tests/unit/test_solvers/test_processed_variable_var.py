#
# Tests for the Processed Variable Var class
#
# This class forms a container for variables (and sensitivities) calculted
#  by the idaklu solver, and does not possesses any capability to calculate
#  values itself since it does not have access to the full state vector
#
from tests import TestCase
import casadi
import pybamm
import tests

import numpy as np
import unittest


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
    var, first_spatial_var, second_spatial_var, disc=None
):
    # first_spatial_var should be on the "smaller" domain, i.e "r" for an "r-x" variable
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
    processed_var = pybamm.ProcessedVariableVar(
        [var_sol],
        [var_casadi],
        [y_sol],
        pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        warn=False,
    )
    # Ordering from idaklu with output_variables set is different to
    # the full solver
    y_sol = y_sol.reshape((y_sol.shape[1], y_sol.shape[0])).transpose()
    np.testing.assert_array_equal(
        processed_var.entries,
        np.reshape(y_sol, [len(first_sol), len(second_sol), len(t_sol)]),
    )
    return y_sol, first_sol, second_sol, t_sol


class TestProcessedVariableVar(TestCase):
    def test_processed_variable_0D(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = y
        var.mesh = None
        t_sol = np.array([0])
        y_sol = np.array([1])[:, np.newaxis]
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariableVar(
            [var],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(processed_var.entries, y_sol[0])

        # check empty sensitivity works

    def test_processed_variable_0D_no_sensitivity(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariableVar(
            [var],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )

        # test no inputs (i.e. no sensitivity)
        self.assertDictEqual(processed_var.sensitivities, {})

        # with parameter
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        a = pybamm.InputParameter("a")
        var = t * y * a
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        inputs = {"a": np.array([1.0])}
        var_casadi = to_casadi(var, y_sol, inputs=inputs)
        processed_var = pybamm.ProcessedVariableVar(
            [var],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), inputs),
            warn=False,
        )

        # test no sensitivity raises error
        assert processed_var.sensitivities is None

    def test_processed_variable_1D(self):
        t = pybamm.t
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = t * var + x

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        sol = pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {})
        processed_var = pybamm.ProcessedVariableVar(
            [var_sol],
            [var_casadi],
            [y_sol],
            sol,
            warn=False,
        )

        # Ordering from idaklu with output_variables set is different to
        # the full solver
        y_sol = y_sol.reshape((y_sol.shape[1], y_sol.shape[0])).transpose()
        np.testing.assert_array_equal(processed_var.entries, y_sol)
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        eqn_casadi = to_casadi(eqn_sol, y_sol)


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
            np.zeros((var_pts[x])),
            "test",
        )

        c = pybamm.StateVector(slice(0, var_pts[x]), domain=["SEI layer"])
        c.mesh = mesh["SEI layer"]
        c_casadi = to_casadi(c, y_sol)
        pybamm.ProcessedVariableVar([c], [c_casadi], [y_sol], solution, warn=False)

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
        processed_var = pybamm.ProcessedVariableVar(
            [var_sol],
            [var_casadi],
            [y_sol],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )

    def test_processed_variable_2D_fixed_t_scikit(self):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y = disc.mesh["current collector"].edges["y"]
        z = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.array([0])
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.ProcessedVariableVar(
            [var_sol],
            [var_casadi],
            [u_sol],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
        )

    def test_3D_raises_error(self):
        var = pybamm.Variable(
            "var",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": ["current collector"]},
        )

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        var_sol = disc.process_symbol(var)
        t_sol = np.array([0, 1, 2])
        u_sol = np.ones(var_sol.shape[0] * 3)[:, np.newaxis]
        var_casadi = to_casadi(var_sol, u_sol)

        with self.assertRaisesRegex(NotImplementedError, "Shape not recognized"):
            pybamm.ProcessedVariableVar(
                [var_sol],
                [var_casadi],
                [u_sol],
                pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
                warn=False,
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
