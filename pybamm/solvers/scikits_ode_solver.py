#
# Solver class using Scipy's adaptive time stepper
#
import casadi
import pybamm

import numpy as np
import importlib
import scipy.sparse as sparse

from .base_solver import validate_max_step

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
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    extra_options : dict, optional
        Any options to pass to the solver.
        Please consult `scikits.odes documentation
        <https://bmcage.github.io/odes/dev/index.html>`_ for details.
        Some common keys:

        - 'linsolver': can be 'dense' (= default), 'lapackdense', 'spgmr', 'spbcgs', \
        'sptfqmr'
    """

    def __init__(
        self,
        method="cvode",
        rtol=1e-6,
        atol=1e-6,
        extrap_tol=None,
        max_step=np.inf,
        extra_options=None,
    ):
        if scikits_odes_spec is None:  # pragma: no cover
            raise ImportError("scikits.odes is not installed")

        super().__init__(method, rtol, atol, extrap_tol=extrap_tol, max_step=max_step)
        self.extra_options = extra_options or {}
        self.ode_solver = True
        self.name = f"Scikits ODE solver ({method})"
        self.max_step = validate_max_step(max_step)

        pybamm.citations.register("Malengier2018")
        pybamm.citations.register("Hindmarsh2000")
        pybamm.citations.register("Hindmarsh2005")

    def _integrate(self, model, t_eval, inputs_dict=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving

        """
        inputs_dict = inputs_dict or {}
        if model.convert_to_format == "casadi":
            inputs = casadi.vertcat(*[x for x in inputs_dict.values()])
        else:
            inputs = inputs_dict

        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full()
        y0 = y0.flatten()

        derivs = model.rhs_eval
        events = model.terminate_events_eval
        jacobian = model.jac_rhs_eval

        if model.convert_to_format == "casadi":

            def eqsydot(t, y, return_ydot):
                return_ydot[:] = derivs(t, y, inputs).full().flatten()

        else:

            def eqsydot(t, y, return_ydot):
                return_ydot[:] = derivs(t, y, inputs).flatten()

        def rootfn(t, y, return_root):
            return_root[:] = [float(event(t, y, inputs)) for event in events]

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0, inputs)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, fy, J):
                    J[:][:] = jacobian(t, y, inputs).toarray()

                def jac_times_vecfn(v, Jv, t, y, userdata):
                    Jv[:] = userdata._jac_eval * v
                    return 0

            else:

                def jacfn(t, y, fy, J):
                    J[:][:] = jacobian(t, y, inputs)

                def jac_times_vecfn(v, Jv, t, y, userdata):
                    Jv[:] = np.matmul(userdata._jac_eval, v)
                    return 0

            def jac_times_setupfn(t, y, fy, userdata):
                userdata._jac_eval = jacobian(t, y, inputs)
                return 0

        extra_options = {
            **self.extra_options,
            "old_api": False,
            "rtol": self.rtol,
            "atol": self.atol,
        }

        # Read linsolver (defaults to dense)
        linsolver = extra_options.get("linsolver", "dense")

        if jacobian:
            if linsolver in ("dense", "lapackdense"):
                extra_options.update({"jacfn": jacfn})
            elif linsolver in ("spgmr", "spbcgs", "sptfqmr"):
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
        timer = pybamm.Timer()
        sol = ode_solver.solve(t_eval, y0)
        integration_time = timer.time()

        # return solution, we need to tranpose y to match scipy's ivp interface
        if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
            if sol.flag == 0:
                termination = "final time"
            # 2 = found root(s)
            elif sol.flag == 2:
                termination = "event"
            if sol.roots.t is None:
                t_root = None
            else:
                t_root = sol.roots.t
            sol = pybamm.Solution(
                sol.values.t,
                np.transpose(sol.values.y),
                model,
                inputs_dict,
                t_root,
                np.transpose(sol.roots.y),
                termination,
            )
            sol.integration_time = integration_time
            return sol
        else:
            raise pybamm.SolverError(sol.message)
