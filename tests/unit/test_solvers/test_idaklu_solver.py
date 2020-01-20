#
# Tests for the KLU Solver class
#
import pybamm
import numpy as np
import scipy.sparse as sparse
import unittest
from tests import get_discretisation_for_testing


@unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
class TestIDAKLUSolver(unittest.TestCase):
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()

        # times and initial conditions
        t_eval = np.linspace(0, 3, 100)
        model.y0 = np.array([0.0, 1.0])

        # Standard pybamm functions
        def jac(t, y):
            J = np.zeros((2, 2))
            J[0][0] = 0.0
            J[0][1] = 1.0
            J[1][0] = 0.0
            J[1][1] = -1.0
            return sparse.csr_matrix(J)

        def event_1(t, y):
            return y[0] - 0.2

        def event_2(t, y):
            return y[1] - 0.0

        model.events_eval = np.array([event_1, event_2])

        def res(t, y, yp):
            # must be of form r = f(t, y) - y'
            r = np.zeros((y.size,))
            r[0] = 0.1 * y[1]
            r[1] = 1 - y[1]
            r[0] += -yp[0]
            return r

        mass_matrix_dense = np.zeros((2, 2))
        mass_matrix_dense[0][0] = 1
        model.mass_matrix = pybamm.Matrix(sparse.csr_matrix(mass_matrix_dense))

        def rhs(t, y):
            return np.array([0.1 * y[1]])

        def alg(t, y):
            return np.array([1 - y[1]])

        solver = pybamm.IDAKLUSolver()
        model.residuals_eval = res
        model.rhs_eval = rhs
        model.algebraic_eval = alg
        model.jacobian_eval = jac

        solution = solver.integrate(model, t_eval)

        # test that final time is time of event
        # y = 0.1 t + y0 so y=0.2 when t=2
        np.testing.assert_array_almost_equal(solution.t[-1], 2.0)

        # test that final value is the event value
        np.testing.assert_array_almost_equal(solution.y[0, -1], 0.2)

        # test that y[1] remains constant
        np.testing.assert_array_almost_equal(
            solution.y[1, :], np.ones(solution.t.shape)
        )

        # test that y[0] = to true solution
        true_solution = 0.1 * solution.t
        np.testing.assert_array_almost_equal(solution.y[0, :], true_solution)

    def test_set_atol(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = pybamm.IDAKLUSolver()

        variable_tols = {"Electrolyte concentration": 1e-3}
        solver.set_atol_by_variable(variable_tols, model)

    def test_model_solver_dae_python(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve
        solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
