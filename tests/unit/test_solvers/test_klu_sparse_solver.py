#
# Tests for the KLU Solver class
#
import pybamm
import numpy as np
import scipy.sparse as sparse
import unittest


@unittest.skipIf(pybamm.have_klu(), "klu is not installed")
class TestKLUSolver(unittest.TestCase):
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf

        # times and initial conditions
        t_eval = np.linspace(0, 3, 100)
        y0 = np.array([0.0, 1.0])

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

        events = np.array([event_1, event_2])

        def res(t, y, yp):
            # must be of form r = f(t, y) - y'
            r = np.zeros((y.size,))
            r[0] = 0.1 * y[1]
            r[1] = 1 - y[1]
            r[0] += -yp[0]
            return r

        mass_matrix_dense = np.zeros((2, 2))
        mass_matrix_dense[0][0] = 1
        mass_matrix = sparse.csr_matrix(mass_matrix_dense)

        def rhs(t, y):
            return np.array([0.1 * y[1]])

        def alg(t, y):
            return np.array([1 - y[1]])

        solver = pybamm.KLU()
        solver.residuals = res
        solver.rhs = rhs
        solver.algebraic = alg

        solution = solver.integrate(res, y0, t_eval, events, mass_matrix, jac)

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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
