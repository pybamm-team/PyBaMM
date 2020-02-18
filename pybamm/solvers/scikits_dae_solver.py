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


class ScikitsDaeSolver(pybamm.BaseSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self,
        method="ida",
        rtol=1e-6,
        atol=1e-6,
        root_method="lm",
        root_tol=1e-6,
        max_steps=1000,
    ):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(method, rtol, atol, root_method, root_tol, max_steps)
        self.name = "Scikits DAE solver ({})".format(method)

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
        residuals = model.residuals_eval
        y0 = model.y0
        events = model.terminate_events_eval
        jacobian = model.jacobian_eval
        mass_matrix = model.mass_matrix.entries

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot)

        def rootfn(t, y, ydot, return_root):
            return_root[:] = [event(t, y) for event in events]

        extra_options = {
            "old_api": False,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, ydot, residuals, cj, J):
                    jac_eval = jacobian(t, y) - cj * mass_matrix
                    J[:][:] = jac_eval.toarray()

            else:

                def jacfn(t, y, ydot, residuals, cj, J):
                    jac_eval = jacobian(t, y) - cj * mass_matrix
                    J[:][:] = jac_eval

            extra_options.update({"jacfn": jacfn})

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        # set up and solve
        dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        sol = dae_solver.solve(t_eval, y0, ydot0)

        # return solution, we need to tranpose y to match scipy's interface
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
