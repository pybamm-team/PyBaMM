#
# Solver class using Scipy's adaptive time stepper
#
import pybamm

import numpy as np
import importlib
import scipy.sparse as sparse

scikits_odes_spec = importlib.util.find_spec("scikits")
if scikits_odes_spec is not None:
    scikits_odes_spec = importlib.util.find_spec("scikits.odes")
    if scikits_odes_spec is not None:
        scikits_odes = importlib.util.module_from_spec(scikits_odes_spec)
        scikits_odes_spec.loader.exec_module(scikits_odes)


def have_scikits_odes():
    return scikits_odes_spec is None


class ScikitsOdeSolver(pybamm.OdeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    linsolver : str, optional
            Can be 'dense' (= default), 'lapackdense', 'spgmr', 'spbcgs', 'sptfqmr'
    """

    def __init__(self, method="cvode", tol=1e-8, linsolver="dense"):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(method, tol)
        self.linsolver = linsolver

    def integrate(
        self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        derivs : method
            A function that takes in t and y and returns the time-derivative dydt
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

        def eqsydot(t, y, return_ydot):
            return_ydot[:] = derivs(t, y)

        def rootfn(t, y, return_root):
            return_root[:] = [event(t, y) for event in events]

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, fy, J):
                    J[:][:] = jacobian(t, y).toarray()

                def jac_times_vecfn(v, Jv, t, y, userdata):
                    Jv[:] = userdata._jac_eval * v
                    return 0

            else:

                def jacfn(t, y, fy, J):
                    J[:][:] = jacobian(t, y)

                def jac_times_vecfn(v, Jv, t, y, userdata):
                    Jv[:] = np.matmul(userdata._jac_eval, v)
                    return 0

            def jac_times_setupfn(t, y, fy, userdata):
                userdata._jac_eval = jacobian(t, y)
                return 0

        extra_options = {
            "old_api": False,
            "rtol": self.tol,
            "atol": self.tol,
            "linsolver": self.linsolver,
        }

        if jacobian:
            if self.linsolver in ("dense", "lapackdense"):
                extra_options.update({"jacfn": jacfn})
            elif self.linsolver in ("spgmr", "spbcgs", "sptfqmr"):
                extra_options.update(
                    {
                        "jac_times_setupfn": jac_times_setupfn,
                        "jac_times_vecfn": jac_times_vecfn,
                        "user_data": self,
                    }
                )

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        ode_solver = scikits_odes.ode(self.method, eqsydot, **extra_options)
        sol = ode_solver.solve(t_eval, y0)

        # return solution, we need to tranpose y to match scipy's ivp interface
        if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
            # 2 = found root(s)
            return sol.values.t, np.transpose(sol.values.y)
        else:
            raise pybamm.SolverError(sol.message)
