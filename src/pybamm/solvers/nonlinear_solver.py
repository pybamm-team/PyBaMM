import numbers
import weakref

import casadi
import numpy as np

import pybamm


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

    def _get_or_build_solver(self, model, inputs_dict):
        """Return a cached C++ StandaloneNewtonSolver, building one if needed."""
        model_id = id(model)

        if model_id in self._solver_cache:
            ref, solver_data = self._solver_cache[model_id]
            if ref() is model:
                return solver_data
            del self._solver_cache[model_id]

        solver_data = self._build_solver(model, inputs_dict)

        def _cleanup(ref, cache=self._solver_cache, key=model_id):
            cache.pop(key, None)

        self._solver_cache[model_id] = (
            weakref.ref(model, _cleanup),
            solver_data,
        )
        return solver_data

    def _build_solver(self, model, inputs_dict):
        """Build CasADi functions and construct the C++ solver.

        Uses model.algebraic_eval (set by BaseSolver.set_up) which has signature
        (t, y_full, p_stacked) -> F_alg.  We rebuild a residual and Jacobian
        that take (t, y_alg, params) where params = [y_diff; inputs].
        """
        from pybammsolvers import idaklu

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

        if len_rhs > 0:
            y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
            y_full = casadi.vertcat(y_diff_sym, y_alg_sym)
            if inputs_len > 0:
                inputs_sym = casadi.MX.sym("inputs", inputs_len)
                p_for_fn = casadi.vertcat(y_diff_sym, inputs_sym)
            else:
                inputs_sym = casadi.MX.sym("inputs", 0)
                p_for_fn = y_diff_sym
        else:
            y_full = y_alg_sym
            if inputs_len > 0:
                inputs_sym = casadi.MX.sym("inputs", inputs_len)
                p_for_fn = inputs_sym
            else:
                inputs_sym = casadi.MX.sym("inputs", 0)
                p_for_fn = casadi.MX.sym("empty", 0)

        p_stacked = casadi.vertcat(inputs_sym) if inputs_len > 0 else casadi.MX(0, 1)

        alg_expr = alg_eval(t_sym, y_full, p_stacked)

        res_fn = casadi.Function(
            "newton_res", [t_sym, y_alg_sym, p_for_fn], [alg_expr]
        )

        jac_expr = casadi.jacobian(alg_expr, y_alg_sym)
        jac_fn = casadi.Function(
            "newton_jac", [t_sym, y_alg_sym, p_for_fn], [jac_expr]
        )

        atol = [float(self.tol)] * len_alg
        rtol = float(self.tol)
        eps_newt = 0.33

        cpp_solver = idaklu.StandaloneNewtonSolver(
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

        pybamm.logger.info(f"Finish building {self.name}")

        return {
            "cpp_solver": cpp_solver,
            "len_rhs": len_rhs,
            "len_alg": len_alg,
        }

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        inputs_dict = inputs_dict or {}

        solver_data = self._get_or_build_solver(model, inputs_dict)
        cpp_solver = solver_data["cpp_solver"]
        len_rhs = solver_data["len_rhs"]
        len_alg = solver_data["len_alg"]

        inputs_flat = np.concatenate(
            [np.atleast_1d(v).ravel() for v in inputs_dict.values()]
        ) if inputs_dict else np.array([], dtype=np.float64)

        y0_np = np.asarray(y0).ravel()
        if len_rhs > 0:
            y0_diff = y0_np[:len_rhs]
            y0_alg = y0_np[len_rhs:]
            p_vec = np.concatenate([y0_diff, inputs_flat])
        else:
            y0_diff = np.array([], dtype=np.float64)
            y0_alg = y0_np
            p_vec = inputs_flat

        y_alg_solutions = [None] * len(t_eval)
        timer = pybamm.Timer()
        integration_time = 0.0

        p_vec = np.ascontiguousarray(p_vec, dtype=np.float64)
        y_alg_guess = np.ascontiguousarray(y0_alg, dtype=np.float64)
        for i, t in enumerate(t_eval):
            timer.reset()
            success, y_alg_arr = cpp_solver.solve(float(t), y_alg_guess, p_vec)
            integration_time += timer.time()

            if not success:
                raise pybamm.SolverError(
                    "Could not find acceptable solution: "
                    "Newton solver did not converge"
                )

            if not np.all(np.isfinite(y_alg_arr)):
                raise pybamm.SolverError(
                    "Could not find acceptable solution: "
                    "solver returned NaNs or Infs"
                )

            y_alg_solutions[i] = y_alg_arr
            y_alg_guess = y_alg_arr

        y_alg_mat = np.column_stack(y_alg_solutions)
        if len_rhs > 0:
            y_diff_mat = np.tile(y0_diff.reshape(-1, 1), (1, len(t_eval)))
            y_sol = np.vstack([y_diff_mat, y_alg_mat])
        else:
            y_sol = y_alg_mat

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
