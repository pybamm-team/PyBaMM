"""Hand-maintained stubs for the compiled extension ``pybamm.rust._core``.

Any change to the Python-visible API in ``packages/pybamm-rust/pybamm-python/src``
must update this file in the same commit; ``mypy.stubtest`` pins names, arities
and defaults against the built extension (``tests/unit/test_rust_stubs.py``),
while the types themselves are review-enforced.
"""

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, TypedDict, final

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

__all__ = [
    "CompiledFunction",
    "CompiledFunctionGroup",
    "CompiledJacobian",
    "CompiledModel",
    "EvaluatorPool",
    "Expr",
    "ExprGraph",
    "PreparedSolver",
    "SolveOutcome",
    "SolverStatistics",
    "_pool_ids",
    "default_solver_options",
]

_FloatArray: TypeAlias = npt.NDArray[np.float64]
# The packed parameter vector, or a {name: value} mapping packed on entry.
_Params: TypeAlias = dict[str, float | _FloatArray] | _FloatArray
# Arguments extracted into Rust vectors: any sequence of floats converts.
_FloatSequence: TypeAlias = Sequence[float] | _FloatArray
_IntSequence: TypeAlias = Sequence[int] | npt.NDArray[np.integer]
# Time grids convert from any 1-D numeric array-like (dtype changes allowed).
_TimeGrid: TypeAlias = (
    Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer]
)

class _SolverOptions(TypedDict):
    """Integrator tuning dict; every key is required (defaults live Python-side)."""

    max_nonlinear_solver_iterations: int
    max_error_test_failures: int
    max_nonlinear_solver_failures: int
    nonlinear_solver_tolerance: float
    min_timestep: float
    max_timestep_growth: float | None
    min_timestep_growth: float | None
    max_timestep_shrink: float | None
    min_timestep_shrink: float | None
    update_jacobian_after_steps: int
    update_rhs_jacobian_after_steps: int
    threshold_to_update_jacobian: float
    threshold_to_update_rhs_jacobian: float
    pi_control_proportional: float
    pi_control_integral: float

class _JacobianStats(TypedDict):
    """Assembly stats returned by :meth:`CompiledModel.jacobian_stats`."""

    strategy: str
    n_colors: int
    nnz: int
    n_dense_rows: int
    n_dense_row_candidates: int
    n_constant_entries: int
    n_swept_columns: int
    jac_lane_width: int
    dense_row_entries: int
    dense_row_tape_instructions: int
    split_eval_primal_instructions: int
    split_eval_total_instructions: int
    split_eval_raw_instructions: int
    split_eval_dispatch_count: int
    branch_block_lens: tuple[int, ...]

@final
class ExprGraph:
    """Arena of expression nodes; the build surface for every compiled artifact."""

    def __new__(cls) -> ExprGraph: ...
    @property
    def n_nodes(self) -> int:
        """Number of nodes in the expression arena."""

    def n_inputs(self) -> int:
        """Total packed width of every registered input (sum of widths, not names)."""

    def scalar(self, value: float) -> Expr: ...
    def time(self) -> Expr: ...
    def state_vector(self, start: int, end: int) -> Expr: ...
    def state_vector_dot(self, start: int, end: int) -> Expr: ...
    def input_parameter(self, name: str, width: int = 1) -> Expr:
        """Register (or re-look-up) a named input; re-registering must repeat the width."""

    def array(self, data: _FloatArray) -> Expr: ...
    def dense_matrix(self, data: _FloatArray, rows: int, cols: int) -> Expr:
        """Dense matrix constant from flat row-major ``data``."""

    def sparse_matrix(
        self,
        indptr: _IntSequence,
        indices: _IntSequence,
        data: _FloatArray,
        rows: int,
        cols: int,
    ) -> Expr:
        """CSR matrix constant."""

    def add(self, a: Expr, b: Expr) -> Expr: ...
    def sub(self, a: Expr, b: Expr) -> Expr: ...
    def mul(self, a: Expr, b: Expr) -> Expr: ...
    def div(self, a: Expr, b: Expr) -> Expr: ...
    def neg(self, a: Expr) -> Expr: ...
    def abs(self, a: Expr) -> Expr: ...
    def pow(self, a: Expr, b: Expr) -> Expr: ...
    def sqrt(self, a: Expr) -> Expr: ...
    def exp(self, a: Expr) -> Expr: ...
    def log(self, a: Expr) -> Expr: ...
    def sin(self, a: Expr) -> Expr: ...
    def cos(self, a: Expr) -> Expr: ...
    def tanh(self, a: Expr) -> Expr: ...
    def sinh(self, a: Expr) -> Expr: ...
    def cosh(self, a: Expr) -> Expr: ...
    def arcsinh(self, a: Expr) -> Expr: ...
    def arctan(self, a: Expr) -> Expr: ...
    def erf(self, a: Expr) -> Expr: ...
    def sign(self, a: Expr) -> Expr: ...
    def floor(self, a: Expr) -> Expr: ...
    def ceiling(self, a: Expr) -> Expr: ...
    def max_reduce(self, a: Expr) -> Expr: ...
    def min_reduce(self, a: Expr) -> Expr: ...
    def matmul(self, a: Expr, b: Expr) -> Expr: ...
    def minimum(self, a: Expr, b: Expr) -> Expr: ...
    def maximum(self, a: Expr, b: Expr) -> Expr: ...
    def modulo(self, a: Expr, b: Expr) -> Expr: ...
    def hypot(self, a: Expr, b: Expr) -> Expr: ...
    def equal_heaviside(self, a: Expr, b: Expr) -> Expr: ...
    def not_equal_heaviside(self, a: Expr, b: Expr) -> Expr: ...
    def equality(self, a: Expr, b: Expr) -> Expr: ...
    def index(self, child: Expr, start: int, end: int) -> Expr: ...
    def concat(self, children: Sequence[Expr]) -> Expr: ...
    def conditional(self, selector: Expr, branches: Sequence[Expr]) -> Expr:
        """Branch ``i`` is active when ``i - 0.5 < selector < i + 0.5`` (1-based)."""

    def interpolant_1d_linear(
        self, x_data: _FloatSequence, y_data: _FloatSequence, child: Expr
    ) -> Expr: ...
    def interpolant_1d_cubic(
        self, breakpoints: _FloatSequence, coeffs: _FloatSequence, child: Expr
    ) -> Expr:
        """``coeffs`` is flat row-major ``[c0..c3]`` groups, one per segment."""

    def interpolant_nd(
        self,
        breakpoints: Sequence[_FloatSequence],
        coeffs: _FloatSequence,
        order: int,
        children: Sequence[Expr],
    ) -> Expr:
        """Tensor-product interpolant over 2 or 3 axes; one child per axis."""

    def eval_to_float(
        self,
        expr: Expr,
        t: float,
        y: _FloatSequence,
        y_dot: _FloatSequence,
        inputs: _FloatSequence,
    ) -> float:
        """Test/debug helper; inputs are not length-validated."""

    def eval_to_array(
        self,
        expr: Expr,
        t: float,
        y: _FloatArray,
        y_dot: _FloatArray,
        inputs: _FloatSequence,
    ) -> _FloatArray:
        """Test/debug helper; inputs are not length-validated."""

    def compile(
        self, expr: Expr, name: str | None = None, n_states: int | None = None
    ) -> CompiledFunction:
        """Compile ``expr`` into an immutable, shareable :class:`CompiledFunction`."""

    def compile_group(
        self,
        outputs: dict[str, Expr],
        name: str | None = None,
        n_states: int | None = None,
    ) -> CompiledFunctionGroup:
        """Compile named outputs into ONE shared tape with cross-output CSE."""

    def dump_dag(
        self, expr: Expr, path: str, model_name: str, n_states: int, n_params: int
    ) -> None:
        """Serialize the graph rooted at ``expr`` to ``path`` (debug snapshot)."""

    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
    def __getnewargs__(self) -> tuple[()]: ...

@final
class Expr:
    """Handle to one node in an :class:`ExprGraph`; only combines within its graph."""

    @property
    def id(self) -> int:
        """Raw node id within the owning graph."""

    def __add__(self, other: Expr, /) -> Expr: ...
    def __sub__(self, other: Expr, /) -> Expr: ...
    def __mul__(self, other: Expr, /) -> Expr: ...
    def __truediv__(self, other: Expr, /) -> Expr: ...
    def __pow__(self, other: Expr, modulo: object = None, /) -> Expr: ...
    def __neg__(self) -> Expr: ...

@final
class CompiledFunction:
    """Prepared evaluation artifact for one expression: eval, JVP and jacobians."""

    def __call__(
        self,
        t: float,
        y: _FloatArray,
        p: _Params,
        y_dot: _FloatArray | None = None,
    ) -> _FloatArray: ...
    def eval(
        self,
        t: float,
        y: _FloatArray,
        p: _Params,
        y_dot: _FloatArray | None = None,
    ) -> _FloatArray:
        """Alias for :meth:`__call__`."""

    def eval_into(
        self,
        t: float,
        y: _FloatArray,
        p: _Params,
        out: _FloatArray,
        y_dot: _FloatArray | None = None,
    ) -> None:
        """Evaluate into pre-allocated ``out`` (length ``output_len``)."""

    def pack(self, mapping: dict[str, float | _FloatArray]) -> _FloatArray:
        """Pack a ``{name: value}`` mapping into the stacked input layout."""

    def jvp(
        self,
        t: float,
        y: _FloatArray,
        p: _Params,
        vy: _FloatArray,
        vp: _FloatArray | None = None,
    ) -> _FloatArray:
        """Forward-mode JVP: ``df/dy @ vy`` (+ ``df/dp @ vp`` when given)."""

    def jacobian(self, wrt: Literal["y", "p"] = "y") -> CompiledJacobian:
        """Lazy, cached-per-``wrt`` prepared jacobian."""

    def eval_trajectory(
        self, ts: _TimeGrid, y_traj: _FloatArray, p: _Params
    ) -> _FloatArray:
        """Evaluate per time column; returns ``(output_len, n_t)`` F-contiguous."""

    def jvp_trajectory(
        self,
        ts: _TimeGrid,
        y_traj: _FloatArray,
        p: _Params,
        vy_traj: _FloatArray,
        vp: _FloatArray | None = None,
    ) -> _FloatArray:
        """Per-column JVP along a trajectory; returns ``(output_len, n_t)``."""

    def eval_trajectory_hermite(
        self,
        t_query: _TimeGrid,
        ts: _TimeGrid,
        ys: _FloatArray,
        yps: _FloatArray,
        p: _Params,
    ) -> _FloatArray:
        """Cubic-Hermite reconstruct the state at ``t_query``, then evaluate."""

    @property
    def input_names(self) -> tuple[str, ...]: ...
    @property
    def n_inputs(self) -> int:
        """Registered-name count (the ``vp`` seed length), NOT the packed width."""

    @property
    def n_states(self) -> int: ...
    @property
    def output_len(self) -> int: ...
    @property
    def uses_y_dot(self) -> bool: ...
    @property
    def name(self) -> str | None: ...
    @property
    def n_instructions(self) -> int:
        """Tape length excluding conditional branch blocks (one dispatch each)."""

    @property
    def n_instructions_total(self) -> int:
        """Raw tape length, branch blocks included."""

    @property
    def n_dispatches(self) -> int: ...
    @property
    def branch_block_lens(self) -> tuple[int, ...]: ...
    @staticmethod
    def _rebuild(
        graph: ExprGraph, root: int, name: str | None, n_states: int | None
    ) -> CompiledFunction: ...
    def __reduce__(
        self,
    ) -> tuple[
        Callable[..., CompiledFunction],
        tuple[ExprGraph, int, str | None, int | None],
    ]: ...

@final
class CompiledFunctionGroup:
    """Named outputs compiled into ONE shared tape; results sliced per output."""

    def __call__(self, t: float, y: _FloatArray, p: _Params) -> list[_FloatArray]: ...
    def eval_trajectory(
        self, ts: _TimeGrid, y_traj: _FloatArray, p: _Params
    ) -> list[_FloatArray]:
        """One ``(output_len_i, n_t)`` F-contiguous array per output, in order."""

    def eval_trajectory_hermite(
        self,
        t_query: _TimeGrid,
        ts: _TimeGrid,
        ys: _FloatArray,
        yps: _FloatArray,
        p: _Params,
    ) -> list[_FloatArray]:
        """Cubic-Hermite reconstruct at ``t_query``, then evaluate and slice."""

    def pack(self, mapping: dict[str, float | _FloatArray]) -> _FloatArray:
        """Pack a ``{name: value}`` mapping into the stacked input layout."""

    @property
    def names(self) -> tuple[str, ...]:
        """Output names in declared order."""

    @property
    def output_lens(self) -> list[int]: ...
    @property
    def input_names(self) -> tuple[str, ...]: ...
    @property
    def n_inputs(self) -> int:
        """Registered-name count (the ``vp`` seed length), NOT the packed width."""

    @property
    def n_states(self) -> int: ...
    @property
    def output_len(self) -> int:
        """Total length across all outputs."""

    @property
    def uses_y_dot(self) -> bool: ...
    @property
    def n_instructions(self) -> int: ...
    @property
    def n_instructions_total(self) -> int: ...
    @property
    def branch_block_lens(self) -> tuple[int, ...]: ...
    @property
    def name(self) -> str | None: ...

@final
class CompiledJacobian:
    """Prepared sparse jacobian: colored JVP sweeps assembled into scipy CSC."""

    def __call__(self, t: float, y: _FloatArray, p: _Params) -> csc_matrix:
        """Assemble and return a ``scipy.sparse.csc_matrix``."""

    def sparsity(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """CSC pattern as ``(indptr, indices)``, cached read-only int32 arrays."""

    @property
    def nnz(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def n_colors(self) -> int: ...
    @property
    def n_dense_rows(self) -> int: ...
    @property
    def wrt(self) -> Literal["y", "p"]: ...

@final
class CompiledModel:
    """Compiled DAE bundle: residual/jacobian FFI surface plus shareable views."""

    @staticmethod
    def from_expr(
        graph: ExprGraph,
        expr: Expr,
        mass_data: _FloatArray,
        mass_indptr: npt.NDArray[np.int64],
        mass_indices: npt.NDArray[np.int64],
        n_inputs: int = 0,
        sens_param_indices: Sequence[int] = ...,
        output_exprs: Sequence[Expr] = ...,
        algebraic_expr: Expr | None = None,
        algebraic_variable_indices: Sequence[int] = ...,
        event_exprs: Sequence[Expr] = ...,
    ) -> CompiledModel:
        """Build from an rhs expression and a CSR mass matrix."""

    @staticmethod
    def _rebuild(
        graph: ExprGraph,
        rhs_root: int,
        output_roots: Sequence[int],
        event_roots: Sequence[int],
        algebraic_root: int | None,
        algebraic_variable_indices: Sequence[int],
        mass_data: Sequence[float],
        mass_indptr: Sequence[int],
        mass_indices: Sequence[int],
        n_inputs: int,
        sens_param_indices: Sequence[int],
    ) -> CompiledModel: ...
    def __reduce__(
        self,
    ) -> tuple[
        Callable[..., CompiledModel],
        tuple[
            ExprGraph,
            int,
            list[int],
            list[int],
            int | None,
            list[int],
            list[float],
            list[int],
            list[int],
            int,
            list[int],
        ],
    ]: ...
    @property
    def rhs(self) -> CompiledFunction:
        """Primal ``f(t, y, p)`` as a shareable view."""

    @property
    def graph(self) -> ExprGraph:
        """The retained derivation arena (observation lowers new roots into it)."""

    @property
    def jacobian(self) -> CompiledJacobian:
        """Pure ``df/dy`` (``cj = 0``, no mass), composed over the bundle's tapes."""

    @property
    def outputs(self) -> list[CompiledFunction]: ...
    @property
    def events(self) -> list[CompiledFunction]: ...
    @property
    def algebraic_residual(self) -> CompiledFunction | None:
        """``g(t, y, p)`` as a shareable view, or ``None`` for ODEs."""

    @property
    def algebraic_jacobian(self) -> CompiledJacobian | None:
        """``dg/dy_alg`` (``n_algebraic`` square), or ``None`` for ODEs."""

    def eval_residual(
        self,
        t: float,
        y: _FloatArray,
        yp: _FloatArray,
        inputs: _FloatArray,
    ) -> _FloatArray:
        """DAE residual ``r = M @ yp - f(t, y)`` as a new array (packed inputs only)."""

    def eval_residual_into(
        self,
        t: float,
        y: _FloatArray,
        yp: _FloatArray,
        inputs: _FloatArray,
        output: _FloatArray,
    ) -> None:
        """DAE residual into pre-allocated ``output`` (length ``n_states``)."""

    def assemble_jacobian_csc_into(
        self,
        t: float,
        y: _FloatArray,
        cj: float,
        inputs: _FloatArray,
        jac_data: _FloatArray,
    ) -> None:
        """Assemble ``df/dy - cj * M`` into a pre-allocated CSC data buffer."""

    def algebraic_ids(self) -> _FloatArray:
        """IDA-convention ids: ``1.0`` differential, ``0.0`` algebraic, per state."""

    def sparsity_pattern(
        self,
    ) -> tuple[npt.NDArray[np.uintp], npt.NDArray[np.uintp]]:
        """CSR ``(indptr, indices)`` of ``df/dy``."""

    def csc_sparsity_pattern(
        self,
    ) -> tuple[npt.NDArray[np.uintp], npt.NDArray[np.uintp]]:
        """CSC ``(colptr, rowind)`` for KLU compatibility."""

    def algebraic_jacobian_sparsity_pattern(
        self,
    ) -> tuple[npt.NDArray[np.uintp], npt.NDArray[np.uintp]]:
        """COO ``(rows, cols)`` of the algebraic jacobian; raises for ODEs."""

    def constant_jacobian_entries(
        self,
    ) -> tuple[npt.NDArray[np.uintp], _FloatArray]:
        """``(csc_idx, value)`` for entries proved constant at compile time."""

    def jacobian_stats(self) -> _JacobianStats: ...
    def evaluator_pool(self, n: int) -> EvaluatorPool:
        """``n`` independent evaluators over this model's tape (``n >= 1``)."""

    @property
    def n_states(self) -> int: ...
    @property
    def n_inputs(self) -> int:
        """Packed input width (core's ``n_params``), matching the C ABI naming."""

    @property
    def has_algebraic(self) -> bool: ...
    @property
    def n_algebraic(self) -> int: ...
    @property
    def algebraic_jacobian_nnz(self) -> int: ...
    @property
    def n_sens_params(self) -> int: ...
    @property
    def n_outputs(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    @property
    def output_len(self) -> int: ...
    @property
    def n_colors(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def jacobian_strategy(self) -> str: ...

@final
class EvaluatorPool:
    """N independent evaluators over one shared tape, one per parallel solver."""

    def as_ptr(self, index: int) -> int:
        """Address of evaluator ``index``; the pool must outlive every use of it.

        Each address is handed out at most once; a second take raises
        ``RuntimeError``, so two solvers can never share one evaluator.
        """

    def __len__(self) -> int: ...

@final
class SolverStatistics:
    """BDF solver statistics for one solve."""

    number_of_steps: int
    number_of_linear_solver_setups: int
    number_of_nonlinear_solver_iterations: int
    number_of_nonlinear_solver_fails: int
    number_of_error_test_failures: int
    number_of_linear_solver_setups_from_checkpoint: int
    number_of_linear_solver_setups_from_first_convergence_fail: int
    number_of_linear_solver_setups_from_second_convergence_fail: int
    number_of_linear_solver_setups_from_error_test_fail: int
    number_of_linear_solver_setups_from_step_success: int
    ic_time_secs: float
    solver_setup_time_secs: float
    integration_time_secs: float
    sens_error_control_relaxed: bool

@final
class SolveOutcome:
    """What one diffsol solve returns, whichever payloads it was asked for."""

    flag: int
    """0 = success, 1 = root found."""
    t_event: float | None
    statistics: SolverStatistics
    @property
    def t(self) -> _FloatArray: ...
    @property
    def y(self) -> _FloatArray:
        """Trajectory, shape ``(n_rows, n_times)``; rows are states, or the
        model's output variables when the solve asked for ``outputs``."""

    @property
    def yp(self) -> _FloatArray | None:
        """Row derivatives ``(n_rows, n_times)``, or ``None`` without
        ``store_yp`` or on an ``outputs`` solve."""

    @property
    def yS(self) -> list[_FloatArray] | None:
        """Per-parameter sensitivities, each matching the flat layout of ``y``,
        or ``None`` when the solve asked for none."""

    @property
    def y_event(self) -> _FloatArray | None:
        """The full state where the trajectory ends, never an outputs row."""

@final
class PreparedSolver:
    """Prepare-once/execute-many diffsol solver for repeated integrations."""

    def __new__(
        cls,
        model: CompiledModel,
        rtol: float = 1e-6,
        atol: float | _FloatArray = ...,
        sens_atol_factor: float = 1e-3,
        options: _SolverOptions | None = None,
        store_yp: bool = False,
    ) -> PreparedSolver: ...
    def solve(
        self,
        t_eval: _FloatArray,
        t_stop: _FloatArray,
        y0: _FloatArray,
        inputs: _FloatArray,
        *,
        outputs: bool = False,
        sensitivities: bool = False,
        y0_sens: _FloatArray | None = None,
    ) -> SolveOutcome:
        """Integrate over ``t_eval``, landing exactly on each ``t_stop``.

        ``outputs`` reports the model's output variables rather than the full
        state; ``sensitivities`` adds the forward-sensitivity blocks, seeded by
        flat ``dy0/dp`` in ``y0_sens``.
        """

    def solve_batch(
        self,
        t_eval: _FloatArray,
        t_stop: _FloatArray,
        y0: _FloatArray,
        inputs: _FloatArray,
        num_threads: int,
        *,
        outputs: bool = False,
        sensitivities: bool = False,
        y0_sens: _FloatArray | None = None,
    ) -> list[SolveOutcome | Exception]:
        """One row of ``y0``/``inputs``/``y0_sens`` per set, answering one
        shared request; a failed set yields its exception instance, unraised."""

def default_solver_options() -> _SolverOptions:
    """Diffsol's own integrator defaults, the dict PyBaMM overlays onto."""

def _pool_ids() -> dict[int, int]:
    """Cached rayon pools as ``{thread_count: pool identity}`` (test introspection)."""
