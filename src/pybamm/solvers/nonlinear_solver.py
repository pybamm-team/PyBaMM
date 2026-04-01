import numbers
import warnings

import casadi
import numpy as np

import pybamm


class NonlinearSolver(pybamm.BaseSolver):
    """Solve algebraic equations using the built-in C++ Newton solver.

    This solver replaces CasADi's rootfinder with a custom Newton solver
    backed by SUNDIALS' KLU (sparse direct) or dense linear solvers.

    Parameters
    ----------
    rtol : float, optional
        Relative tolerance for the WRMS norm (default 1e-6).
    atol : float or array-like, optional
        Absolute tolerance for the WRMS norm (default 1e-6).
        Can be a scalar or per-variable array.
    step_tol : float, optional
        Step size tolerance (default 1e-6). Newton iteration stops
        when both the WRMS norm converges and the step norm <= step_tol.
    options : dict, optional
        Options for the nonlinear solver:

        - ``"jacobian"``: ``"sparse"`` (default) or ``"dense"``
        - ``"linear_solver"``: ``"SUNLinSol_KLU"`` (default) or ``"SUNLinSol_Dense"``
        - ``"max_iterations"``: max Newton iterations (default 100)
        - ``"max_linesearch_backtracks"``: max line search backtracks (default 100)
        - ``"epsNewt"``: WRMS convergence threshold (default 0.0033)

    on_failure : str, optional
        What to do if the solver fails: ``"raise"`` (default), ``"warn"``,
        or ``"ignore"``.
    """

    def __init__(self, rtol=1e-6, atol=1e-6, step_tol=1e-6, options=None, on_failure=None):
        super().__init__(rtol=rtol, atol=atol, on_failure=on_failure)
        self.step_tol = step_tol
        self._algebraic_solver = True
        self._options = options or {}
        self.name = "Newton algebraic solver"

    @property
    def algebraic_solver(self):
        return True

    def set_up_root_solver(self, model, inputs_dict, t_eval):
        """Build CasADi functions and create the C++ NonlinearSolver.

        The residual function F(t, x, p) is constructed such that:
        - x = y_alg (the algebraic unknowns)
        - p = [y_diff; inputs] (the parameters, held fixed during solve)
        - F evaluates the algebraic part of the model equations

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose algebraic equations to solve.
        inputs_dict : dict
            Dictionary of input parameters.
        t_eval : array-like
            Time points (not used during setup).
        """
        pybamm.logger.info(f"Start building {self.name}")

        from pybammsolvers import idaklu

        y0 = model.y0_list[0]

        if model.rhs == {}:
            len_rhs = 0
        elif model.len_rhs_and_alg == y0.shape[0]:
            len_rhs = model.len_rhs
        else:
            len_rhs = model.len_rhs + model.len_rhs_sens

        len_alg = y0.shape[0] - len_rhs

        # Build symbolic variables
        t_sym = casadi.MX.sym("t")
        x_sym = casadi.MX.sym("x", len_alg)

        # Parameters: [y_diff, inputs]
        y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
        inputs_casadi = {}
        for name, value in inputs_dict.items():
            if isinstance(value, numbers.Number):
                inputs_casadi[name] = casadi.MX.sym(name)
            else:
                inputs_casadi[name] = casadi.MX.sym(name, value.shape[0])
        inputs_sym = casadi.vertcat(*[p for p in inputs_casadi.values()])
        p_sym = casadi.vertcat(y_diff_sym, inputs_sym)

        # Full state vector: [y_diff; x]
        y_sym = casadi.vertcat(y_diff_sym, x_sym)

        # Algebraic residual: F(t, x, p)
        alg = model.casadi_algebraic(t_sym, y_sym, inputs_sym)
        residual_fn = casadi.Function("F", [t_sym, x_sym, p_sym], [alg])

        # Jacobian of F w.r.t. x
        J = casadi.jacobian(alg, x_sym)
        jacobian_fn = casadi.Function("J", [t_sym, x_sym, p_sym], [J])

        # Serialize and create C++ objects
        res_cpp = idaklu.generate_function(residual_fn.serialize())
        jac_cpp = idaklu.generate_function(jacobian_fn.serialize())

        # Build atol vector for the algebraic variables
        model_atol = getattr(model, "atol", self.atol)
        model_atol = self._check_atol_type(model_atol, model)
        if isinstance(model_atol, np.ndarray) and len(model_atol) > len_alg:
            alg_atol = model_atol[len_rhs:]
        elif isinstance(model_atol, np.ndarray):
            alg_atol = model_atol
        else:
            alg_atol = np.full(len_alg, float(model_atol))

        inputs_length = int(inputs_sym.shape[0]) + len_rhs

        model.nonlinear_root_solver = idaklu.create_nonlinear_solver(
            res_cpp,
            jac_cpp,
            len_alg,
            alg_atol,
            float(self.rtol),
            float(self.step_tol),
            inputs_length,
            self._options,
        )
        model._nonlinear_len_rhs = len_rhs
        model._nonlinear_len_alg = len_alg

        pybamm.logger.info(f"Finish building {self.name}")

    def _check_atol_type(self, atol, model):
        """Ensure atol is the right type and shape."""
        if isinstance(atol, (float, int)):
            return float(atol)
        atol = np.asarray(atol, dtype=float).ravel()
        return atol

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        """Solve algebraic equations -- thin dispatch to C++.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose algebraic equations to solve.
        t_eval : array-like
            Time points at which to solve.
        inputs_dict : dict
            Input parameter dictionary.
        y0 : array-like
            Full initial state vector [y_diff; y_alg].

        Returns
        -------
        :class:`pybamm.Solution`
        """
        if getattr(model, "nonlinear_root_solver", None) is None:
            self.set_up_root_solver(model, inputs_dict, t_eval)

        len_rhs = model._nonlinear_len_rhs

        y0 = np.asarray(y0, dtype=np.float64).ravel()
        y0_diff = y0[:len_rhs]
        y0_alg = y0[len_rhs:]

        if inputs_dict:
            inputs_flat = np.concatenate(
                [np.atleast_1d(v).ravel() for v in inputs_dict.values()]
            )
        else:
            inputs_flat = np.array([], dtype=np.float64)
        # Parameters to C++: [y_diff, inputs]
        p = np.concatenate([y0_diff, inputs_flat])

        t_eval = np.asarray(t_eval, dtype=np.float64)

        timer = pybamm.Timer()
        result = model.nonlinear_root_solver.solve(t_eval, y0_alg, p, None)
        integration_time = timer.time()

        if not result["success"]:
            msg = f"Could not find acceptable solution: {result['message']}"
            match self._on_failure:
                case "raise":
                    raise pybamm.SolverError(msg)
                case "warn":
                    warnings.warn(msg + ", returning a partial solution.", stacklevel=2)

        # Reconstruct full y: [y_diff; x] for each time point
        x_sol = result["x"]  # shape (len_alg, n_times)
        n_times = x_sol.shape[1]
        y_diff_repeated = np.tile(y0_diff.reshape(-1, 1), (1, n_times))
        y_sol = np.vstack([y_diff_repeated, x_sol])

        sol = pybamm.Solution(
            [t_eval[:n_times]],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
        )
        sol.integration_time = integration_time
        return sol
