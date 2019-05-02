#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
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


class ScikitsOdeSolver(pybamm.OdeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, method="cvode", tol=1e-8):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(tol)
        self._method = method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    def integrate(
        self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None,
        linsolver = 'dense'
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
        linsolver : method, optional
            Can be 'dense' (= default), 'lapackdense', 'spgmr', 'spbcgs', 'sptfqmr'

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

        extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol,
                "linsolver": linsolver}

        if jacobian:
            if linsolver in ('dense', 'lapackdense'):
                extra_options.update({
                    "jacfn": jacfn
                })
            elif linsolver in ('spgmr', 'spbcgs', 'sptfqmr'):
                extra_options.update({
                    "jac_times_setupfn": jac_times_setupfn,
                    "jac_times_vecfn": jac_times_vecfn,
                    "user_data": self
                })

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        ode_solver = scikits_odes.ode(self.method, eqsydot, **extra_options)
        sol = ode_solver.solve(t_eval, y0)

        # return solution, we need to tranpose y to match scipy's ivp interface
        return sol.values.t, np.transpose(sol.values.y)
