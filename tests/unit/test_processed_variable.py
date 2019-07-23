#
# Tests for the Processed Variable class
#
import pybamm
import tests

import numpy as np
import unittest


class TestProcessedVariable(unittest.TestCase):
    def test_processed_variable_1D(self):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        processed_var = pybamm.ProcessedVariable(var, t_sol, y_sol)
        np.testing.assert_array_equal(processed_var.entries, t_sol * y_sol[0])

    def test_processed_variable_2D(self):
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

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(processed_var.entries[1:-1], y_sol)
        np.testing.assert_array_equal(processed_var(t_sol, x_sol), y_sol)
        processed_eqn = pybamm.ProcessedVariable(eqn_sol, t_sol, y_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )

        # Test extrapolation
        np.testing.assert_array_equal(processed_var.entries[0], 2 * y_sol[0] - y_sol[1])
        np.testing.assert_array_equal(
            processed_var.entries[1], 2 * y_sol[-1] - y_sol[-2]
        )

        # On edges
        x_s_edge = pybamm.Matrix(disc.mesh["separator"][0].edges, domain="separator")
        processed_x_s_edge = pybamm.ProcessedVariable(x_s_edge, t_sol, y_sol, disc.mesh)
        np.testing.assert_array_equal(
            x_s_edge.entries[:, 0], processed_x_s_edge.entries[1:-1, 0]
        )

    def test_processed_variable_3D(self):
        var = pybamm.Variable("var", domain=["negative particle"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable("r", domain=["negative particle"])

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(x_sol), len(r_sol), len(t_sol)]),
        )

    @unittest.skipIf(pybamm.have_scikit_fem(), "scikit-fem not installed")
    def test_processed_variable_3D_scikit(self):
        var = pybamm.Variable("var", domain=["current collector"])
        y = pybamm.SpatialVariable("y", domain=["current collector"])
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.process_symbol(y).entries[:, 0]
        z_sol = disc.process_symbol(z).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, u_sol, mesh=disc.mesh)
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(u_sol, [len(y_sol), len(z_sol), len(t_sol)]),
        )

    def test_processed_var_1D_interpolation(self):
        # without spatial dependence
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = t * y

        t_sol = np.linspace(0, 1, 1000)
        y_sol = np.array([np.linspace(0, 5, 1000)])
        processed_var = pybamm.ProcessedVariable(var, t_sol, y_sol)
        # vector
        np.testing.assert_array_equal(processed_var(t_sol), y_sol[0])
        # scalar
        np.testing.assert_array_equal(processed_var(0.5), 2.5)
        np.testing.assert_array_equal(processed_var(0.7), 3.5)

        processed_eqn = pybamm.ProcessedVariable(eqn, t_sol, y_sol)
        np.testing.assert_array_equal(processed_eqn(t_sol), t_sol * y_sol[0])
        np.testing.assert_array_almost_equal(processed_eqn(0.5), 0.5 * 2.5)

    def test_processed_var_2D_interpolation(self):
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

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        # 2 vectors
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        # 1 vector, 1 scalar
        np.testing.assert_array_almost_equal(
            processed_var(0.5, x_sol)[:, 0], 2.5 * x_sol
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol[-1]), x_sol[-1] * np.linspace(0, 5)
        )
        # 2 scalars
        np.testing.assert_array_almost_equal(
            processed_var(0.5, x_sol[-1]), 2.5 * x_sol[-1]
        )
        processed_eqn = pybamm.ProcessedVariable(eqn_sol, t_sol, y_sol, mesh=disc.mesh)
        # 2 vectors
        np.testing.assert_array_almost_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )
        # 1 vector, 1 scalar
        self.assertEqual(processed_eqn(0.5, x_sol[10:30]).shape, (20, 1))
        self.assertEqual(processed_eqn(t_sol[4:9], x_sol[-1]).shape, (5,))
        # 2 scalars
        self.assertEqual(processed_eqn(0.5, x_sol[-1]).shape, (1,))

        # On microscale
        r_n = pybamm.Matrix(
            disc.mesh["negative particle"][0].nodes, domain="negative particle"
        )
        processed_r_n = pybamm.ProcessedVariable(r_n, t_sol, y_sol, disc.mesh)
        np.testing.assert_array_equal(r_n.entries[:, 0], processed_r_n.entries[1:-1, 0])
        np.testing.assert_array_almost_equal(
            processed_r_n(0, r=np.linspace(0, 1))[:, 0], np.linspace(0, 1)
        )

    def test_processed_var_3D_interpolation(self):
        var = pybamm.Variable("var", domain=["negative particle"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable("r", domain=["negative particle"])

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (40, 10, 50)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol),
            np.reshape(y_sol, [len(x_sol), len(r_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, x_sol, r_sol).shape, (40, 10))
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
        x = pybamm.SpatialVariable("x", domain=["positive electrode"])
        r = pybamm.SpatialVariable("r", domain=["positive particle"])

        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, y_sol, mesh=disc.mesh)
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (35, 10, 50)
        )

    @unittest.skipIf(pybamm.have_scikit_fem(), "scikit-fem not installed")
    def test_processed_var_3D_scikit_interpolation(self):
        var = pybamm.Variable("var", domain=["current collector"])
        y = pybamm.SpatialVariable("y", domain=["current collector"])
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.process_symbol(y).entries[:, 0]
        z_sol = disc.process_symbol(z).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)

        processed_var = pybamm.ProcessedVariable(var_sol, t_sol, u_sol, mesh=disc.mesh)
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, y_sol, z_sol).shape, (15, 15, 50)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, y_sol, z_sol),
            np.reshape(u_sol, [len(y_sol), len(z_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, y_sol, z_sol).shape, (15, 15))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, z_sol).shape, (15, 50))
        np.testing.assert_array_equal(processed_var(t_sol, y_sol, 0.5).shape, (15, 50))
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, 0.2, z_sol).shape, (15,))
        np.testing.assert_array_equal(processed_var(0.5, y_sol, 0.5).shape, (15,))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, 0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, 0.2, 0.2).shape, ())

    def test_processed_variable_ode_pde_solution(self):
        # without space
        model = pybamm.BaseBatteryModel()
        c = pybamm.Variable("conc")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c}
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        t_sol, y_sol = modeltest.solution.t, modeltest.solution.y
        processed_vars = pybamm.post_process_variables(model.variables, t_sol, y_sol)
        np.testing.assert_array_almost_equal(processed_vars["c"](t_sol), np.exp(-t_sol))

        # with space
        # set up and solve model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseBatteryModel()
        c = pybamm.Variable("conc", domain=whole_cell)
        c_s = pybamm.Variable("particle conc", domain="negative particle")
        model.rhs = {c: -c, c_s: 1 - c_s}
        model.initial_conditions = {c: 1, c_s: 0.5}
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")},
            c_s: {"left": (0, "Neumann"), "right": (0, "Neumann")},
        }
        model.variables = {
            "c": c,
            "N": pybamm.grad(c),
            "c_s": c_s,
            "N_s": pybamm.grad(c_s),
        }
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
        # set up testing
        t_sol, y_sol = modeltest.solution.t, modeltest.solution.y
        x = pybamm.SpatialVariable("x", domain=whole_cell)
        x_sol = modeltest.disc.process_symbol(x).entries[:, 0]
        processed_vars = pybamm.post_process_variables(
            model.variables, t_sol, y_sol, modeltest.disc.mesh
        )

        # test
        np.testing.assert_array_almost_equal(
            processed_vars["c"](t_sol, x_sol),
            np.ones_like(x_sol)[:, np.newaxis] * np.exp(-t_sol),
        )

    def test_failure(self):
        t = np.ones(25)
        y = np.ones((120, 25))
        mat = pybamm.Vector(np.ones(120), domain=["negative particle"])
        disc = tests.get_p2d_discretisation_for_testing()
        with self.assertRaisesRegex(
            ValueError, "3D variable shape does not match domain shape"
        ):
            pybamm.ProcessedVariable(mat, t, y, disc.mesh)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
