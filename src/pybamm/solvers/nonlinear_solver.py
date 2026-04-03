import numbers
import weakref

import casadi
import numpy as np

import pybamm
from pybammsolvers import idaklu


class NonlinearSolver(pybamm.BaseSolver):
    """Solve a discretised model containing only (time independent) algebraic
    equations using the C++ Newton solver backed by SUNDIALS KLU/Dense.

    Drop-in replacement for :class:`CasadiAlgebraicSolver` with significantly
    lower per-solve overhead: CasADi functions are evaluated via the C API and
    the linear system is solved by KLU (sparse) or LAPACK (dense), all in a
    single C++ call with no Python round-trips per Newton iteration.

    Parameters
    ----------
    tol : float, optional
        Absolute tolerance for the residual (default 1e-6).
    step_tol : float, optional
        Tolerance on the Newton step norm (default 1e-4).
    max_iter : int, optional
        Maximum Newton iterations (default 100).
    max_backtracks : int, optional
        Maximum Armijo linesearch backtracks per iteration (default 5).
    use_sparse : bool, optional
        Use KLU sparse factorisation (True, default) or dense LAPACK (False).
    """

    def __init__(
        self,
        tol=1e-6,
        step_tol=1e-4,
        max_iter=100,
        max_backtracks=5,
        use_sparse=True,
    ):
        super().__init__()
        self.tol = tol
        self.step_tol = step_tol
        self.max_iter = max_iter
        self.max_backtracks = max_backtracks
        self.use_sparse = use_sparse
        self.name = "Newton algebraic solver"
        self._algebraic_solver = True
        self._solver_cache = {}

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    @property
    def step_tol(self):
        return self._step_tol

    @step_tol.setter
    def step_tol(self, value):
        self._step_tol = value

    def set_up_root_solver(self, model, inputs_dict, t_eval):
        """Build CasADi functions and construct the C++ solver."""
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

        inputs_len = sum(
            1 if isinstance(v, numbers.Number) else np.asarray(v).size
            for v in inputs_dict.values()
        ) if inputs_dict else 0

        y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
        inputs_sym = casadi.MX.sym("inputs", inputs_len)
        y_full = casadi.vertcat(y_diff_sym, y_alg_sym)
        # The parameter vector stacks the differential states and inputs
        p_stacked = casadi.vertcat(y_diff_sym, inputs_sym)

        alg_expr = alg_eval(t_sym, y_full, p_stacked)

        res_fn = casadi.Function(
            "newton_res", [t_sym, y_alg_sym, p_stacked], [alg_expr]
        )

        jac_expr = casadi.jacobian(alg_expr, y_alg_sym)
        jac_fn = casadi.Function(
            "newton_jac", [t_sym, y_alg_sym, p_stacked], [jac_expr]
        )

        atol = [float(self.tol)] * len_alg
        rtol = float(self.tol)
        eps_newt = 0.33

        root_solver = idaklu.StandaloneNewtonSolver(
            idaklu.generate_function(res_fn.serialize()),
            idaklu.generate_function(jac_fn.serialize()),
            atol,
            rtol,
            float(self.step_tol),
            self.max_iter,
            self.max_backtracks,
            eps_newt,
            self.use_sparse,
        )

        model.algebraic_root_solver = root_solver

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        inputs_dict = inputs_dict or {}

        root_solver = getattr(model, "algebraic_root_solver", None)
        if root_solver is None:
            self.set_up_root_solver(model, inputs_dict, t_eval)
            root_solver = model.algebraic_root_solver
        len_rhs = model.len_rhs

        inputs_flat = np.concatenate(
            [np.atleast_1d(v).ravel() for v in inputs_dict.values()]
        ) if inputs_dict else np.array([], dtype=np.float64)

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

        _check_success(success, y_alg_mat)

        y_diff_mat = np.tile(y0_diff.reshape(-1, 1), (1, len(t_eval)))
        y_sol = np.vstack([y_diff_mat, y_alg_mat])

        sol = pybamm.Solution(
            [t_eval],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
            all_t_evals=[t_eval],
        )
        sol.integration_time = integration_time
        return sol

def _check_success(success: bool, y_alg_mat: np.ndarray):
    if not np.isfinite(np.linalg.norm(y_alg_mat, ord=np.inf)):
        raise pybamm.SolverError(
            "Could not find acceptable solution: "
            "solver returned NaNs or Infs"
        )
    if not success:
        raise pybamm.SolverError(
            "Could not find acceptable solution: "
            "Newton solver did not converge"
        )