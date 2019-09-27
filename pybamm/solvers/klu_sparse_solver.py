#
# Solver class using sundials with the KLU sparse linear solver
#
import pybamm

import numpy as np
from .c_solvers import klu
import scipy.sparse as sparse


class KLU(pybamm.DaeSolver):
    """Solve a discretised model, using sundials with the KLU sparse linear solver.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    tolerance : float, optional
        The tolerance for the initial-condition solver (default is 1e-8).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self, method="ida", tol=1e-8, root_method="lm", root_tol=1e-6, max_steps=1000
    ):

        super().__init__(method, tol, root_method, root_tol, max_steps)

    def integrate(
        self, residuals, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        """

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot)

        rtol = self.tol
        atol = self.tol

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, cj):
                    return jacobian(t, y) - cj * mass_matrix

            else:

                def jacfn(t, y, cj):
                    jac_eval = jacobian(t, y) - cj * mass_matrix
                    return sparse.csr_matrix(jac_eval)

        # just defining this here for now...
        class SundialsJacobian:
            def __init__(self):
                self.J = None

                J = jacfn(0, y0, 0.1)
                self.nnz = J.nnz  # hoping nnz remains constant...

            def jac_res(self, t, y, cj):
                # must be of form j_res = (dr/dy) - (cj) (dr/dy')
                # cj is just the input parameter
                # see p68 of the ida_guide.pdf for more details
                self.J = jacfn(t, y, cj)

            def get_jac_data(self):
                return self.J.data

            def get_jac_row_vals(self):
                return self.J.indices

            def get_jac_col_ptrs(self):
                return self.J.indptr

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        jac_class = SundialsJacobian()

        num_of_events = len(events)
        use_jac = 1

        def rootfn(t, y):
            return_root = np.ones((num_of_events,))
            return_root[:] = [event(t, y) for event in events]

            return return_root

        # solve
        sol = klu.solve(
            t_eval,
            y0,
            ydot0,
            self.residuals,
            jac_class.jac_res,
            jac_class.get_jac_data,
            jac_class.get_jac_row_vals,
            jac_class.get_jac_col_ptrs,
            jac_class.nnz,
            rootfn,
            num_of_events,
            use_jac,
            mass_matrix.diagonal(),  # this should always just be the diagonals
            rtol,
            atol,
        )

        t = sol.t
        number_of_timesteps = t.size
        number_of_states = y0.size
        y_out = sol.y.reshape((number_of_timesteps, number_of_states))

        # return solution, we need to tranpose y to match scipy's interface
        if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
            if sol.flag == 0:
                termination = "final time"
            # 2 = found root(s)
            elif sol.flag == 2:
                termination = "event"
            return pybamm.Solution(
                sol.t, np.transpose(y_out), t[-1], np.transpose(y_out[-1]), termination
            )
        else:
            raise pybamm.SolverError(sol.message)
