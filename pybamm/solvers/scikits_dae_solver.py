#
# Solver class using Scipy's adaptive time stepper
#
import casadi
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
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        If a solver class, must be an algebraic solver class.
        If "casadi",
        the solver uses casadi's Newton rootfinding algorithm to find initial
        conditions. Otherwise, the solver uses 'scipy.optimize.root' with method
        specified by 'root_method' (e.g. "lm", "hybr", ...)
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not (default is 0).
    extra_options : dict, optional
        Any options to pass to the solver.
        Please consult `scikits.odes documentation
        <https://bmcage.github.io/odes/dev/index.html>`_ for details.
        Some common keys:

        - 'max_steps': maximum (int) number of steps the solver can take
    """

    def __init__(
        self,
        method="ida",
        rtol=1e-6,
        atol=1e-6,
        root_method="casadi",
        root_tol=1e-6,
        extrap_tol=0,
        extra_options=None,
        max_steps="deprecated",
    ):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(
            method, rtol, atol, root_method, root_tol, extrap_tol, max_steps
        )
        self.name = "Scikits DAE solver ({})".format(method)

        self.extra_options = extra_options or {}

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
            y0 = y0.full().flatten()

        residuals = model.residuals_eval
        events = model.terminate_events_eval
        jacobian = model.jacobian_eval
        mass_matrix = model.mass_matrix.entries

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot, inputs)

        def rootfn(t, y, ydot, return_root):
            return_root[:] = [event(t, y, inputs) for event in events]

        extra_options = {
            **self.extra_options,
            "old_api": False,
            "rtol": self.rtol,
            "atol": self.atol,
        }

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0, inputs)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, ydot, residuals, cj, J):
                    jac_eval = jacobian(t, y, inputs) - cj * mass_matrix
                    J[:][:] = jac_eval.toarray()

            else:

                def jacfn(t, y, ydot, residuals, cj, J):
                    jac_eval = jacobian(t, y, inputs) - cj * mass_matrix
                    J[:][:] = jac_eval

            extra_options.update({"jacfn": jacfn})

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        # set up and solve
        dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        timer = pybamm.Timer()
        sol = dae_solver.solve(t_eval, y0, ydot0)
        integration_time = timer.time()

        # return solution, we need to tranpose y to match scipy's interface
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
