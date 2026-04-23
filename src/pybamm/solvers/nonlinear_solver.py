import warnings

import casadi
import numpy as np
from pybammsolvers import idaklu

import pybamm
from pybamm.codegen.compilation import aot_compile

_DEFAULT_OPTIONS = {
    "compile": False,
}


class NonlinearSolver(pybamm.BaseSolver):
    """Solve a discretised model containing only (time independent) algebraic
    equations using the C++ Newton solver backed by SUNDIALS KLU/Dense.

    Drop-in replacement for :class:`CasadiAlgebraicSolver` with significantly
    lower per-solve overhead: CasADi functions are evaluated via the C API and
    the linear system is solved by KLU (sparse) or LAPACK (dense), all in a
    single C++ call with no Python round-trips per Newton iteration.

    Parameters
    ----------
    atol : float, optional
        Absolute tolerance for the algebraic variables (default 1e-6).
    rtol : float, optional
        Relative tolerance for the algebraic variables (default 1e-4).
    step_tol : float, optional
        Tolerance on the Newton step norm (default 1e-4).
    max_iter : int, optional
        Maximum Newton iterations (default 100).
    max_backtracks : int, optional
        Maximum Armijo linesearch backtracks per iteration (default 5).
    eps_newt : float, optional
        WRMS convergence coefficient (default 0.33).
    use_sparse : bool, optional
        Use KLU sparse factorisation (True, default) or dense LAPACK (False).
    on_failure : str, optional
        Behaviour on convergence failure (default "error").
    options : dict, optional
        Solver options. Currently supports ``compile`` (bool, default False)
        for ahead-of-time compilation of CasADi residual/Jacobian.
    """

    def __init__(
        self,
        atol=1e-6,
        rtol=1e-4,
        step_tol=1e-4,
        max_iter=100,
        max_backtracks=5,
        eps_newt=0.33,
        use_sparse=True,
        on_failure=None,
        options=None,
    ):
        super().__init__()
        self.atol = atol
        self.rtol = rtol
        self.step_tol = step_tol
        self.max_iter = max_iter
        self.max_backtracks = max_backtracks
        self.eps_newt = eps_newt
        self.use_sparse = use_sparse
        self.name = "Nonlinear solver"
        self.on_failure = on_failure or "error"
        self._algebraic_solver = True
        self._user_options = options or {}
        self._options = _DEFAULT_OPTIONS | self._user_options

    @staticmethod
    def _check_tolerance(value):
        if value < 0:
            raise ValueError("Tolerance must be non-negative")

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, value):
        self._check_tolerance(value)
        self._atol = value

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, value):
        self._check_tolerance(value)
        self._rtol = value

    @property
    def step_tol(self):
        return self._step_tol

    @step_tol.setter
    def step_tol(self, value):
        self._check_tolerance(value)
        self._step_tol = value

    def set_up_root_solver(self, model, inputs_dict, t_eval):
        """Build CasADi functions and construct the C++ solver.

        The resulting ``StandaloneNewtonSolver`` is attached to the model as
        ``model.algebraic_root_solver``. This mirrors the layout used by
        :class:`CasadiAlgebraicSolver` so different models in the same
        experiment (e.g. each unique step model) get independently sized
        Newton solvers without one stomping on another's cache.
        """
        pybamm.logger.info(f"Start building {self.name}")

        y0 = model.y0_list[0]
        if model.rhs == {}:
            len_rhs = 0
        elif model.len_rhs_and_alg == y0.shape[0]:
            len_rhs = model.len_rhs
        else:
            len_rhs = model.len_rhs + model.len_rhs_sens

        len_alg = y0.shape[0] - len_rhs

        alg_eval = model.algebraic_eval

        t_sym = casadi.MX.sym("t")
        y_alg_sym = casadi.MX.sym("y_alg", len_alg)

        inputs_len = (
            sum(np.asarray(v).size for v in inputs_dict.values()) if inputs_dict else 0
        )

        y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
        inputs_sym = casadi.MX.sym("inputs", inputs_len)
        y_full = casadi.vertcat(y_diff_sym, y_alg_sym)
        # The parameter vector stacks the differential states and inputs
        p_stacked = casadi.vertcat(y_diff_sym, inputs_sym)

        alg_expr = alg_eval(t_sym, y_full, inputs_sym)

        res_fn = casadi.Function(
            "newton_res", [t_sym, y_alg_sym, p_stacked], [alg_expr]
        )

        jac_expr = casadi.jacobian(alg_expr, y_alg_sym)
        jac_fn = casadi.Function(
            "newton_jac", [t_sym, y_alg_sym, p_stacked], [jac_expr]
        )

        if self._options["compile"]:
            res_fn, jac_fn = aot_compile([res_fn, jac_fn])

        model.algebraic_root_solver = self._build_newton_solver(res_fn, jac_fn, len_alg)

    def _build_newton_solver(self, res_fn, jac_fn, len_alg):
        return idaklu.StandaloneNewtonSolver(
            residual=idaklu.generate_function(res_fn.serialize()),
            jacobian=idaklu.generate_function(jac_fn.serialize()),
            atol=np.full(len_alg, float(self.atol)).tolist(),
            rtol=float(self.rtol),
            step_tol=float(self.step_tol),
            max_iter=int(self.max_iter),
            max_backtracks=int(self.max_backtracks),
            eps_newt=float(self.eps_newt),
            use_sparse=bool(self.use_sparse),
        )

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        inputs_dict = inputs_dict or {}

        if getattr(model, "algebraic_root_solver", None) is None:
            self.set_up_root_solver(model, inputs_dict, t_eval)
        root_solver = model.algebraic_root_solver
        len_rhs = model.len_rhs

        inputs_flat = (
            np.concatenate([np.atleast_1d(v).ravel() for v in inputs_dict.values()])
            if inputs_dict
            else np.empty(0)
        )

        y0_np = np.asarray(y0).ravel()
        y0_diff = y0_np[:len_rhs]
        y0_alg = np.ascontiguousarray(y0_np[len_rhs:], dtype=np.float64)
        p_vec = np.ascontiguousarray(
            np.concatenate([y0_diff, inputs_flat]), dtype=np.float64
        )
        t_eval_np = np.ascontiguousarray(t_eval, dtype=np.float64)

        timer = pybamm.Timer()
        success, y_alg_mat = root_solver.solve_batch(t_eval_np, y0_alg, p_vec)
        integration_time = timer.time()

        self._check_success(success, y_alg_mat)

        y_diff_mat = np.tile(y0_diff.reshape(-1, 1), (1, len(t_eval)))
        y_sol = np.vstack([y_diff_mat, y_alg_mat])

        sol = pybamm.Solution(
            [t_eval],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
            all_t_evals=[t_eval],
            options=self._options,
        )
        sol.integration_time = integration_time
        return sol

    def _check_success(self, success: bool, y_alg_mat: np.ndarray):
        if self.on_failure == "ignore":
            return

        messages = []
        if not np.isfinite(np.linalg.norm(y_alg_mat, ord=np.inf)):
            messages.append("Solver returned NaNs or Infs")
        if not success:
            messages.append("Newton solver did not converge")

        if not messages:
            return

        message = "Could not find acceptable solution:" + "\n".join(messages)
        if self.on_failure == "warn":
            warnings.warn(message, pybamm.SolverWarning, stacklevel=2)
        else:
            raise pybamm.SolverError(message)
