#
# Algebraic solver class
#
import casadi
import pybamm
import numpy as np
from scipy import optimize
from scipy.sparse import issparse


class AlgebraicSolver(pybamm.BaseSolver):
    """Solve a discretised model which contains only (time independent) algebraic
    equations using a root finding algorithm.
    Uses scipy.optimize.root.
    Note: this solver could be extended for quasi-static models, or models in
    which the time derivative is manually discretised and results in a (possibly
    nonlinear) algebaric system at each time level.

    Parameters
    ----------
    method : str, optional
        The method to use to solve the system (default is "lm"). If it starts with
        "lsq", least-squares minimization is used. The method for least-squares can be
        specified in the form "lsq_methodname"
    tol : float, optional
        The tolerance for the solver (default is 1e-6).
    extra_options : dict, optional
        Any options to pass to the rootfinder. Vary depending on which method is chosen.
        Please consult `SciPy documentation <https://tinyurl.com/ybr6cfqs>`_ for
        details.
    """

    def __init__(self, method="lm", tol=1e-6, extra_options=None):
        super().__init__(method=method)
        self.tol = tol
        self.extra_options = extra_options or {}
        self.name = "Algebraic solver ({})".format(method)
        self.algebraic_solver = True
        pybamm.citations.register("Virtanen2020")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    def _integrate(self, model, t_eval, inputs_dict=None):
        """
        Calculate the solution of the algebraic equations through root-finding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
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

        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        # Split y0 into differential and algebraic
        if model.rhs == {}:
            len_rhs = 0
        else:
            len_rhs = model.rhs_eval(t_eval[0], y0, inputs).shape[0]
        y0_diff, y0_alg = np.split(y0, [len_rhs])

        algebraic = model.algebraic_eval

        y_alg = np.empty((len(y0_alg), len(t_eval)))

        timer = pybamm.Timer()
        integration_time = 0
        for idx, t in enumerate(t_eval):

            def root_fun(y_alg):
                "Evaluates algebraic using y"
                y = np.concatenate([y0_diff, y_alg])
                out = algebraic(t, y, inputs)
                pybamm.logger.debug(
                    "Evaluating algebraic equations at t={}, L2-norm is {}".format(
                        t * model.timescale_eval, np.linalg.norm(out)
                    )
                )
                return out

            jac = model.jac_algebraic_eval
            if jac:
                if issparse(jac(t_eval[0], y0, inputs)):

                    def jac_fn(y_alg):
                        """
                        Evaluates jacobian using y0_diff (fixed) and y_alg (varying)
                        """
                        y = np.concatenate([y0_diff, y_alg])
                        return jac(0, y, inputs)[:, len_rhs:].toarray()

                else:

                    def jac_fn(y_alg):
                        """
                        Evaluates jacobian using y0_diff (fixed) and y_alg (varying)
                        """
                        y = np.concatenate([y0_diff, y_alg])
                        return jac(0, y, inputs)[:, len_rhs:]

            else:
                jac_fn = None

            # Evaluate algebraic with new t and previous y0, if it's already close
            # enough then keep it
            if np.all(abs(algebraic(t, y0, inputs)) < self.tol):
                pybamm.logger.debug("Keeping same solution at t={}".format(t))
                y_alg[:, idx] = y0_alg
            # Otherwise calculate new y0
            else:
                # Methods which use least-squares are specified as either "lsq", which
                # uses the default method, or with "lsq__methodname"
                if self.method.startswith("lsq"):

                    if self.method == "lsq":
                        method = "trf"
                    else:
                        method = self.method[5:]
                    if jac_fn is None:
                        jac_fn = "2-point"
                    timer.reset()
                    sol = optimize.least_squares(
                        root_fun,
                        y0_alg,
                        method=method,
                        ftol=self.tol,
                        jac=jac_fn,
                        bounds=model.bounds,
                        **self.extra_options,
                    )
                    integration_time += timer.time()
                # Methods which use minimize are specified as either "minimize", which
                # uses the default method, or with "minimize__methodname"
                elif self.method.startswith("minimize"):
                    # Adapt the root function for minimize
                    def root_norm(y):
                        return np.sum(root_fun(y) ** 2)

                    if jac_fn is None:
                        jac_norm = None
                    else:

                        def jac_norm(y):
                            return np.sum(2 * root_fun(y) * jac_fn(y), 0)

                    if self.method == "minimize":
                        method = None
                    else:
                        method = self.method[10:]
                    extra_options = self.extra_options
                    if np.any(model.bounds[0] != -np.inf) or np.any(
                        model.bounds[1] != np.inf
                    ):
                        bounds = [
                            (lb, ub) for lb, ub in zip(model.bounds[0], model.bounds[1])
                        ]
                        extra_options["bounds"] = bounds
                    timer.reset()
                    sol = optimize.minimize(
                        root_norm,
                        y0_alg,
                        method=method,
                        tol=self.tol,
                        jac=jac_norm,
                        **extra_options,
                    )
                    integration_time += timer.time()
                else:
                    timer.reset()
                    sol = optimize.root(
                        root_fun,
                        y0_alg,
                        method=self.method,
                        tol=self.tol,
                        jac=jac_fn,
                        options=self.extra_options,
                    )
                    integration_time += timer.time()

                if sol.success and np.all(abs(sol.fun) < self.tol):
                    # update initial guess for the next iteration
                    y0_alg = sol.x
                    # update solution array
                    y_alg[:, idx] = y0_alg
                elif not sol.success:
                    raise pybamm.SolverError(
                        "Could not find acceptable solution: {}".format(sol.message)
                    )
                else:
                    raise pybamm.SolverError(
                        "Could not find acceptable solution: solver terminated "
                        "successfully, but maximum solution error "
                        "({}) above tolerance ({})".format(
                            np.max(abs(sol.fun)), self.tol
                        )
                    )

        # Concatenate differential part
        y_diff = np.r_[[y0_diff] * len(t_eval)].T
        y_sol = np.r_[y_diff, y_alg]
        # Return solution object (no events, so pass None to t_event, y_event)
        sol = pybamm.Solution(t_eval, y_sol, model, inputs_dict, termination="success")
        sol.integration_time = integration_time
        return sol
