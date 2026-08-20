import warnings

import casadi
import numpy as np
from pybammsolvers import idaklu

import pybamm
from pybamm.codegen.compilation import aot_compile
from pybamm.solvers.base_solver import flatten_inputs
from pybamm.solvers.rust_lowering import RustModelLowering

_DEFAULT_OPTIONS = {
    "compile": False,
}


class _NonlinearSolverSetup:
    """Pickle-safe wrapper around StandaloneNewtonSolver"""

    __slots__ = ["_keepalive", "_setup"]

    def __init__(self, setup: idaklu.StandaloneNewtonSolver, keepalive=None):
        self._setup = setup
        self._keepalive = keepalive  # EvaluatorPool backing the C++ raw ptr

    def __bool__(self):
        # falsy once the handle is gone -> caller rebuilds, never reuses a
        # dangling pointer (e.g. after a pickle round-trip)
        return self._setup is not None

    def __getstate__(self):
        return {"_setup": None, "_keepalive": None}

    def __setstate__(self, state):
        self._setup = None
        self._keepalive = None

    def solve_batch(self, *args, **kwargs):
        return self._setup.solve_batch(*args, **kwargs)


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

    def __getstate__(self):
        # _model_set_up holds unpicklable rust artifacts, so clear it: the solver
        # stays picklable and the next solve() rebuilds from the model.
        state = self.__dict__.copy()
        state["_model_set_up"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model_set_up = {}

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

    def _set_up_root_solver(self, model, inputs_dict, t_eval):
        """Build CasADi functions and construct the C++ solver.

        The resulting ``StandaloneNewtonSolver`` is attached to the model as
        ``model.algebraic_root_solver``. This mirrors the layout used by
        :class:`CasadiAlgebraicSolver` so different models in the same
        experiment (e.g. each unique step model) get independently sized
        Newton solvers without one stomping on another's cache.
        """
        if model.convert_to_format == "rust":
            return self._set_up_root_solver_rust(model, inputs_dict)

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

        return self._build_newton_solver(res_fn, jac_fn, len_alg)

    def _set_up_root_solver_rust(self, model, inputs_dict):
        """Build the Rust-backed standalone Newton solver.

        Lowers rhs+algebraic into one CompiledModel (same layout as the
        IDAKLU rust set-up) and hands its algebraic sub-block to the C++
        Newton driver via the dlsym FFI.
        """
        pybamm.logger.info(f"Start building {self.name} (rust)")
        if self._options["compile"]:
            raise pybamm.SolverError(
                "options['compile'] is CasADi-only; not supported with "
                "convert_to_format='rust'"
            )
        len_rhs = 0 if model.rhs == {} else model.len_rhs
        # An oversized consistent y0 means the rhs/alg block was extended with
        # sensitivities, which the Rust Newton driver does not support.
        y0_list = getattr(model, "y0_list", None)
        if y0_list and model.len_rhs_and_alg != np.asarray(y0_list[0]).shape[0]:
            raise pybamm.SolverError(
                "The Rust Newton root solver does not support "
                "sensitivity-extended states; use convert_to_format='casadi' "
                "for this configuration"
            )

        lowering = RustModelLowering(model, inputs_dict)
        lowering.state_residual(algebraic_only=len_rhs == 0)
        lowering.algebraic_block(first_algebraic_index=len_rhs)
        rust_model = lowering.compile()
        jac_rows, jac_cols = rust_model.algebraic_jacobian_sparsity_pattern()
        len_alg = model.len_rhs_and_alg - len_rhs
        # algebraic_jacobian_sparsity_pattern returns global state columns, but the C++ Newton
        # builds an n_alg x n_alg system, so localise to the algebraic block.
        jac_cols = [int(c) - len_rhs for c in np.asarray(jac_cols)]
        jac_rows = np.asarray(jac_rows).tolist()
        if any(c < 0 or c >= len_alg for c in jac_cols):
            raise pybamm.SolverError(
                "Rust algebraic jacobian has columns outside the algebraic "
                f"block [0, {len_alg}); cannot localise for the Newton solver"
            )

        # The C++ Newton driver mutates the evaluator it drives, so its address
        # must come from the pool's exclusive handout, not a rust_model borrow.
        pool = rust_model.evaluator_pool(1)
        _setup = idaklu.StandaloneNewtonSolver(
            rust_model=pool.as_ptr(0),
            n_rhs=len_rhs,
            n_alg=len_alg,
            jac_rows=jac_rows,
            jac_cols=jac_cols,
            atol=np.full(len_alg, float(self.atol)).tolist(),
            rtol=float(self.rtol),
            step_tol=float(self.step_tol),
            max_iter=int(self.max_iter),
            max_backtracks=int(self.max_backtracks),
            eps_newt=float(self.eps_newt),
            use_sparse=bool(self.use_sparse),
        )
        return _NonlinearSolverSetup(_setup, keepalive=pool)

    def _build_newton_solver(self, res_fn, jac_fn, len_alg):
        _setup = idaklu.StandaloneNewtonSolver(
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
        return _NonlinearSolverSetup(_setup)

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        inputs_dict = inputs_dict or {}

        root_solver = self.get_root_solver(model, inputs_dict, t_eval)
        len_rhs = model.len_rhs

        inputs_flat = flatten_inputs(inputs_dict)

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
