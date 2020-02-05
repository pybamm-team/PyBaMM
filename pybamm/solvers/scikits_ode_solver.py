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
    return scikits_odes_spec is not None


class ScikitsOdeSolver(pybamm.BaseSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    linsolver : str, optional
            Can be 'dense' (= default), 'lapackdense', 'spgmr', 'spbcgs', 'sptfqmr'
    """

    def __init__(self, method="cvode", rtol=1e-6, atol=1e-6, linsolver="dense"):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(method, rtol, atol)
        self.linsolver = linsolver
        self.ode_solver = True
        self.name = "Scikits ODE solver ({})".format(method)

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        """
        derivs = model.rhs_eval
        y0 = model.y0
        events = model.terminate_events_eval
        jacobian = model.jacobian_eval

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
            "rtol": self.rtol,
            "atol": self.atol,
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
            if sol.flag == 0:
                termination = "final time"
            # 2 = found root(s)
            elif sol.flag == 2:
                termination = "event"
            return pybamm.Solution(
                sol.values.t,
                np.transpose(sol.values.y),
                sol.roots.t,
                np.transpose(sol.roots.y),
                termination,
            )
        else:
            raise pybamm.SolverError(sol.message)
