#
# Tests for the Processed Variable class
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
    processed_var = pybamm.ProcessedVariable(
        [var_sol],
        [var_casadi],
        pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        warn=False,
    )
    np.testing.assert_array_equal(
        processed_var.entries,
        np.reshape(y_sol, [len(first_sol), len(second_sol), len(t_sol)]),
    )
    return y_sol, first_sol, second_sol, t_sol


class TestProcessedVariable(TestCase):
    def test_processed_variable_0D(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(processed_var.entries, t_sol * y_sol[0])

        # scalar value
        var = y
        var.mesh = None
        t_sol = np.array([0])
        y_sol = np.array([1])[:, np.newaxis]
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var],
            [var_casadi],
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
        processed_var = pybamm.ProcessedVariable(
            [var],
            [var_casadi],
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
        processed_var = pybamm.ProcessedVariable(
            [var],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), inputs),
            warn=False,
        )

        # test no sensitivity raises error
        with self.assertRaisesRegex(ValueError, "Cannot compute sensitivities"):
            print(processed_var.sensitivities)

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
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(processed_var.entries, y_sol)
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn = pybamm.ProcessedVariable(
            [eqn_sol],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_almost_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )

        # Test extrapolation
        np.testing.assert_array_equal(processed_var.entries[0], 2 * y_sol[0] - y_sol[1])
        np.testing.assert_array_equal(
            processed_var.entries[1], 2 * y_sol[-1] - y_sol[-2]
        )

        # On edges
        x_s_edge = pybamm.Matrix(disc.mesh["separator"].edges, domain="separator")
        x_s_edge.mesh = disc.mesh["separator"]
        x_s_casadi = to_casadi(x_s_edge, y_sol)
        processed_x_s_edge = pybamm.ProcessedVariable(
            [x_s_edge],
            [x_s_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            x_s_edge.entries[:, 0], processed_x_s_edge.entries[:, 0]
        )

        # space only
        eqn = var + x
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.array([0])
        y_sol = np.ones_like(x_sol)[:, np.newaxis]
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn2 = pybamm.ProcessedVariable(
            [eqn_sol],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_eqn2.entries, y_sol + x_sol[:, np.newaxis]
        )

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
        pybamm.ProcessedVariable([c], [c_casadi], solution, warn=False)

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
        process_and_check_2D_variable(var, R, x, disc=disc)

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
        process_and_check_2D_variable(var, R, z, disc=disc)

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
        process_and_check_2D_variable(var, r, R, disc=disc)

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
        x_s_edge.mesh = disc.mesh["separator"]
        x_s_edge.secondary_mesh = disc.mesh["current collector"]
        x_s_casadi = to_casadi(x_s_edge, y_sol)
        processed_x_s_edge = pybamm.ProcessedVariable(
            [x_s_edge],
            [x_s_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
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
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )

    def test_processed_variable_2D_scikit(self):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y = disc.mesh["current collector"].edges["y"]
        z = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
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
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
        )

    def test_processed_var_0D_interpolation(self):
        # without spatial dependence
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = t * y
        var.mesh = None
        eqn.mesh = None

        t_sol = np.linspace(0, 1, 1000)
        y_sol = np.array([np.linspace(0, 5, 1000)])
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # vector
        np.testing.assert_array_equal(processed_var(t_sol), y_sol[0])
        # scalar
        np.testing.assert_array_equal(processed_var(0.5), 2.5)
        np.testing.assert_array_equal(processed_var(0.7), 3.5)

        eqn_casadi = to_casadi(eqn, y_sol)
        processed_eqn = pybamm.ProcessedVariable(
            [eqn],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(processed_eqn(t_sol), t_sol * y_sol[0])
        np.testing.assert_array_almost_equal(processed_eqn(0.5), 0.5 * 2.5)

        # Suppress warning for this test
        pybamm.set_logging_level("ERROR")
        np.testing.assert_array_equal(processed_eqn(2), np.nan)
        pybamm.set_logging_level("WARNING")

    def test_processed_var_0D_fixed_t_interpolation(self):
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = 2 * y
        var.mesh = None
        eqn.mesh = None

        t_sol = np.array([10])
        y_sol = np.array([[100]])
        eqn_casadi = to_casadi(eqn, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [eqn],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )

        np.testing.assert_array_equal(processed_var(), 200)

    def test_processed_var_1D_interpolation(self):
        t = pybamm.t
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = t * var + x

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.linspace(0, 1)
        y_sol = x_sol[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 2 vectors
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        # 1 vector, 1 scalar
        np.testing.assert_array_almost_equal(processed_var(0.5, x_sol), 2.5 * x_sol)
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol[-1]), x_sol[-1] * np.linspace(0, 5)
        )
        # 2 scalars
        np.testing.assert_array_almost_equal(
            processed_var(0.5, x_sol[-1]), 2.5 * x_sol[-1]
        )
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn = pybamm.ProcessedVariable(
            [eqn_sol],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 2 vectors
        np.testing.assert_array_almost_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )
        # 1 vector, 1 scalar
        self.assertEqual(processed_eqn(0.5, x_sol[10:30]).shape, (20,))
        self.assertEqual(processed_eqn(t_sol[4:9], x_sol[-1]).shape, (5,))
        # 2 scalars
        self.assertEqual(processed_eqn(0.5, x_sol[-1]).shape, ())

        # test x
        x_disc = disc.process_symbol(x)
        x_casadi = to_casadi(x_disc, y_sol)

        processed_x = pybamm.ProcessedVariable(
            [x_disc],
            [x_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_almost_equal(processed_x(t=0, x=x_sol), x_sol)

        # In particles
        r_n = pybamm.Matrix(
            disc.mesh["negative particle"].nodes, domain="negative particle"
        )
        r_n.mesh = disc.mesh["negative particle"]
        r_n_casadi = to_casadi(r_n, y_sol)
        processed_r_n = pybamm.ProcessedVariable(
            [r_n],
            [r_n_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(r_n.entries[:, 0], processed_r_n.entries[:, 0])
        r_test = np.linspace(0, 0.5)
        np.testing.assert_array_almost_equal(processed_r_n(0, r=r_test), r_test)

        # On size domain
        R_n = pybamm.Matrix(
            disc.mesh["negative particle size"].nodes, domain="negative particle size"
        )
        R_n.mesh = disc.mesh["negative particle size"]
        R_n_casadi = to_casadi(R_n, y_sol)
        processed_R_n = pybamm.ProcessedVariable(
            [R_n],
            [R_n_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        np.testing.assert_array_equal(R_n.entries[:, 0], processed_R_n.entries[:, 0])
        R_test = np.linspace(0, 1)
        np.testing.assert_array_almost_equal(processed_R_n(0, R=R_test), R_test)

    def test_processed_var_1D_fixed_t_interpolation(self):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = var + x

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.array([1])
        y_sol = x_sol[:, np.newaxis]

        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [eqn_sol],
            [eqn_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )

        # vector
        np.testing.assert_array_almost_equal(
            processed_var(x=x_sol), 2 * x_sol[:, np.newaxis]
        )
        # scalar
        np.testing.assert_array_almost_equal(processed_var(x=0.5), 1)

    def test_processed_var_2D_interpolation(self):
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
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 40, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x_sol, r_sol),
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, x_sol, r_sol).shape, (10, 40))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, r_sol).shape, (10, 50))
        np.testing.assert_array_equal(processed_var(t_sol, x_sol, 0.5).shape, (40, 50))
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, 0.2, r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(0.5, x_sol, 0.5).shape, (40,))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, 0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, 0.2, 0.2).shape, ())

        # positive particle
        var = pybamm.Variable(
            "var",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["positive electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )

        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 35, 50)
        )

    def test_processed_var_2D_fixed_t_interpolation(self):
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
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 2 vectors
        np.testing.assert_array_equal(
            processed_var(t=0, x=x_sol, r=r_sol).shape, (10, 40)
        )
        # 1 vector, 1 scalar
        np.testing.assert_array_equal(processed_var(t=0, x=0.2, r=r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(t=0, x=x_sol, r=0.5).shape, (40,))
        # 2 scalars
        np.testing.assert_array_equal(processed_var(t=0, x=0.2, r=0.2).shape, ())

    def test_processed_var_2D_secondary_broadcast(self):
        var = pybamm.Variable("var", domain=["negative particle"])
        broad_var = pybamm.SecondaryBroadcast(var, "negative electrode")
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable("r", domain=["negative particle"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(broad_var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 40, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x_sol, r_sol),
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, x_sol, r_sol).shape, (10, 40))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, r_sol).shape, (10, 50))
        np.testing.assert_array_equal(processed_var(t_sol, x_sol, 0.5).shape, (40, 50))
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, 0.2, r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(0.5, x_sol, 0.5).shape, (40,))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, 0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, 0.2, 0.2).shape, ())

        # positive particle
        var = pybamm.Variable("var", domain=["positive particle"])
        broad_var = pybamm.SecondaryBroadcast(var, "positive electrode")
        x = pybamm.SpatialVariable("x", domain=["positive electrode"])
        r = pybamm.SpatialVariable("r", domain=["positive particle"])

        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(broad_var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 35, 50)
        )

    def test_processed_var_2D_scikit_interpolation(self):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.mesh["current collector"].edges["y"]
        z_sol = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, y=y_sol, z=z_sol).shape, (15, 15, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, y=y_sol, z=z_sol),
            np.reshape(u_sol, [len(y_sol), len(z_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(
            processed_var(0.5, y=y_sol, z=z_sol).shape, (15, 15)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, y=0.2, z=z_sol).shape, (15, 50)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, y=y_sol, z=0.5).shape, (15, 50)
        )
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, y=0.2, z=z_sol).shape, (15,))
        np.testing.assert_array_equal(processed_var(0.5, y=y_sol, z=0.5).shape, (15,))
        np.testing.assert_array_equal(processed_var(t_sol, y=0.2, z=0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, y=0.2, z=0.2).shape, ())

    def test_processed_var_2D_fixed_t_scikit_interpolation(self):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.mesh["current collector"].edges["y"]
        z_sol = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.array([0])
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.ProcessedVariable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, u_sol, pybamm.BaseModel(), {}),
            warn=False,
        )
        # 2 vectors
        np.testing.assert_array_equal(
            processed_var(t=0, y=y_sol, z=z_sol).shape, (15, 15)
        )
        # 1 vector, 1 scalar
        np.testing.assert_array_equal(processed_var(t=0, y=0.2, z=z_sol).shape, (15,))
        np.testing.assert_array_equal(processed_var(t=0, y=y_sol, z=0.5).shape, (15,))
        # 2 scalars
        np.testing.assert_array_equal(processed_var(t=0, y=0.2, z=0.2).shape, ())

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
            pybamm.ProcessedVariable(
                [var_sol],
                [var_casadi],
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
