"""Per-variable observation strategies for :class:`pybamm.ProcessedVariable`.

A :class:`VariableObserver` holds one variable's per-sub-solution leaves and
evaluates them on the solution grid, off-grid via cubic Hermite, and through
the forward chain rule for sensitivities.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence

import casadi
import numpy as np
import numpy.typing as npt
from pybammsolvers import idaklu

import pybamm


class SegmentSelector:
    """Which sub-solutions cover a set of query times.

    The non-empty segments and their end times are found once, on
    construction, rather than on every query.

    Parameters
    ----------
    all_ts : list of numpy.ndarray
        Per-sub-solution time arrays, successively increasing.
    """

    def __init__(self, all_ts: Sequence[npt.NDArray[np.float64]]):
        self.indices = np.where([ti.size > 0 for ti in all_ts])[0]
        self._starts = [all_ts[idx][0] for idx in self.indices]
        self._ends = [all_ts[idx][-1] for idx in self.indices]

    def select(
        self, t: npt.NDArray[np.float64], full_range: bool
    ) -> npt.NDArray[np.intp]:
        """Indices into ``all_ts`` of the segments covering ``t``.

        Parameters
        ----------
        t : numpy.ndarray
            Sorted query times.
        full_range : bool
            Whether to keep every non-empty segment, rather than only those
            whose span contains at least one of ``t`` (plus the last segment
            when ``t`` extends past it).

        Returns
        -------
        numpy.ndarray
            Segment indices, ascending. Empty segments are always dropped.
        """
        if full_range:
            return self.indices
        return self.indices[_find_ts_indices(self._starts, self._ends, t)]


class VariableObserver(ABC):
    """How one variable's leaves are evaluated over a solution's segments.

    Implementations read only these attributes of the ``variable`` handed to
    them: ``all_ts``, ``all_ys``, ``all_yps``, ``all_inputs``,
    ``all_inputs_stacked``, ``t_pts``, ``hermite_interpolation``,
    ``time_integral``, ``base_variables``, ``sensitivity_names``,
    ``all_solution_sensitivities``, ``_name`` and ``_shape``.
    """

    #: Derived caches: built on demand, never pickled (see __getstate__).
    _selector = None
    _selector_ts = None
    _serialised = None

    def __getstate__(self):
        """The observer's state without its derived caches, rebuilt on demand."""
        state = self.__dict__.copy()
        for key in ("_selector", "_selector_ts", "_serialised"):
            state.pop(key, None)
        return state

    @property
    @abstractmethod
    def leaves(self) -> list:
        """The variable's per-sub-solution evaluable leaves, in solve order."""

    def segments(
        self,
        variable: pybamm.ProcessedVariable,
        t: npt.NDArray[np.float64],
        full_range: bool,
    ) -> npt.NDArray[np.intp]:
        """Indices of ``variable``'s sub-solutions covering ``t``.

        See :meth:`SegmentSelector.select`.
        """
        # An observer can be shared across solutions, so key on their time arrays
        if self._selector_ts is not variable.all_ts:
            self._selector = SegmentSelector(variable.all_ts)
            self._selector_ts = variable.all_ts
        return self._selector.select(t, full_range)

    @abstractmethod
    def observe_raw(
        self, variable: pybamm.ProcessedVariable
    ) -> npt.NDArray[np.float64]:
        """Evaluate on the solution's own time points, shaped by ``_shape``."""

    @abstractmethod
    def observe_hermite(
        self, variable: pybamm.ProcessedVariable, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate at arbitrary sorted times ``t``, cubic-Hermite in state."""

    @abstractmethod
    def sensitivities(
        self, variable: pybamm.ProcessedVariable
    ) -> dict[str, npt.NDArray[np.float64]]:
        """``{"all": (N, n_p), param: (N,)}`` forward sensitivities of the variable."""


class CasadiObserver(VariableObserver):
    """Observation through serialised CasADi functions and the IDAKLU kernels.

    Parameters
    ----------
    leaves : list of :class:`casadi.Function`
        One function per sub-solution, evaluating ``(t, y, p_stacked)``.
    """

    def __init__(self, leaves: list):
        self._leaves = leaves

    @property
    def leaves(self) -> list:
        return self._leaves

    def _serialise(self, idxs):
        """Serialised leaves for `idxs`, memoised by leaf identity.

        CasADi functions are immutable and serialising one is about half the
        cost of an observe call, so the bytes are built once per leaf.
        """
        if self._serialised is None:
            self._serialised = {}
        serialised = self._serialised
        funcs = [None] * len(idxs)
        for i, idx in enumerate(idxs):
            leaf = self._leaves[idx]
            key = id(leaf)
            if key not in serialised:
                serialised[key] = leaf.serialize()
            funcs[i] = serialised[key]
        return funcs

    def _setup(self, variable, t, full_range):
        """Per-segment IDAKLU inputs: ``(ts, ys, yps, funcs, inputs, is_f_contiguous)``."""
        pybamm.logger.debug("Setting up C++ interpolation inputs")
        idxs = self.segments(variable, t, full_range)
        hermite = variable.hermite_interpolation
        all_ts, all_ys = variable.all_ts, variable.all_ys

        ts = [all_ts[idx] for idx in idxs]
        ys = [all_ys[idx] for idx in idxs]
        yps = [variable.all_yps[idx] for idx in idxs] if hermite else None
        inputs = [variable.all_inputs_stacked[idx] for idx in idxs]

        is_f_contiguous = _is_f_contiguous(ys)

        ts = idaklu.VectorRealtypeNdArray(ts)
        ys = idaklu.VectorRealtypeNdArray(ys)
        yps = idaklu.VectorRealtypeNdArray(yps) if hermite else None
        inputs = idaklu.VectorRealtypeNdArray(inputs)

        return ts, ys, yps, self._serialise(idxs), inputs, is_f_contiguous

    def observe_raw(self, variable):
        pybamm.logger.debug("Observing the variable raw data")
        t = variable.t_pts
        ts, ys, _, funcs, inputs, is_f_contiguous = self._setup(
            variable, t, full_range=True
        )
        return idaklu.observe(
            ts, ys, inputs, funcs, is_f_contiguous, variable._shape(t)
        )

    def observe_hermite(self, variable, t):
        pybamm.logger.debug("Observing and Hermite interpolating the variable")
        ts, ys, yps, funcs, inputs, _ = self._setup(variable, t, full_range=False)
        return idaklu.observe_hermite_interp(
            t, ts, ys, yps, inputs, funcs, variable._shape(t)
        )

    def sensitivities(self, variable):
        sensitivity_names = variable.sensitivity_names
        all_S_var = []
        for ts, ys, inputs, base_variable, dy_dp in zip(
            variable.all_ts,
            variable.all_ys,
            variable.all_inputs,
            variable.base_variables,
            variable.all_solution_sensitivities["all"],
            strict=True,
        ):
            sensitivity_inputs = {
                name: inputs[name] for name in sensitivity_names if name in inputs
            }
            sensitivity_inputs_stacked = casadi.vertcat(
                *[sensitivity_inputs[name] for name in sensitivity_names]
            )

            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in sensitivity_inputs.items()
            }

            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Symbolic for the sensitivity inputs only; the others keep their values
            inputs_for_casadi = {**inputs, **p_casadi}

            var_casadi = base_variable.to_casadi(
                t_casadi, y_casadi, inputs=inputs_for_casadi
            )
            dvar_dy = casadi.jacobian(var_casadi, y_casadi)
            dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

            dvar_dy_func = casadi.Function(
                "dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy]
            )
            dvar_dp_func = casadi.Function(
                "dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp]
            )
            dvar_dy_eval = casadi.diagcat(
                *[
                    dvar_dy_func(t, ys[:, idx], sensitivity_inputs_stacked)
                    for idx, t in enumerate(ts)
                ]
            )
            dvar_dp_eval = casadi.vertcat(
                *[
                    dvar_dp_func(t, ys[:, idx], sensitivity_inputs_stacked)
                    for idx, t in enumerate(ts)
                ]
            )

            S_var = dvar_dy_eval @ dy_dp + dvar_dp_eval

            all_S_var.append(S_var)

        S_var = np.vstack(all_S_var)
        if variable.time_integral is not None:
            # The value sums over every segment at once, so its sensitivity must too
            S_var = variable.time_integral.postfix_sensitivities(
                variable._name,
                self.observe_raw(variable),
                variable.t_pts,
                variable.all_inputs[0],
                sensitivity_names,
                S_var,
            )
        return pack_sensitivity_dict(S_var, sensitivity_names)


class NativeObserver(VariableObserver):
    """Observation through compiled Rust tapes lowered into the retained graph.

    ``placeholder_states`` holds each segment's state count for outputs-only
    solves, which store no states; ``None`` when states are real.
    """

    def __init__(self, leaves, placeholder_states=None, post_sum_leaf=None):
        self._leaves = leaves
        self._placeholder_states = placeholder_states
        self._post_sum_leaf = post_sum_leaf

    @property
    def leaves(self):
        return self._leaves

    def _setup(self, variable, t, full_range):
        """Per-segment ``(ts, ys, yps, inputs, leaves)``; inputs stay dicts."""
        idxs = self.segments(variable, t, full_range)
        ts = [variable.all_ts[idx] for idx in idxs]
        if self._placeholder_states is None:
            ys = [variable.all_ys[idx] for idx in idxs]
        else:
            ys = [
                np.zeros((self._placeholder_states[idx], ti.size))
                for idx, ti in zip(idxs, ts, strict=True)
            ]
        yps = (
            [variable.all_yps[idx] for idx in idxs]
            if variable.hermite_interpolation
            else None
        )
        inputs = [variable.all_inputs[idx] for idx in idxs]
        leaves = [self._leaves[idx] for idx in idxs]
        return ts, ys, yps, inputs, leaves

    def observe_raw(self, variable):
        # eval_trajectory returns (output_len, n_t), so reshape in order="F"
        t = variable.t_pts
        ts, ys, _, inputs, leaves = self._setup(variable, t, full_range=True)
        cols = [
            np.asarray(leaf.eval_trajectory(t_i, y_i, inp_i))
            for leaf, t_i, y_i, inp_i in zip(leaves, ts, ys, inputs, strict=True)
        ]
        return np.concatenate(cols, axis=1).reshape(variable._shape(t), order="F")

    def observe_hermite(self, variable, t):
        # Route query times to segments as observe.cpp's knot-window scan does
        ts, ys, yps, inputs, leaves = self._setup(variable, t, full_range=False)
        n_segments = len(ts)
        cols = []
        i = 0
        n = len(t)
        for seg_idx, (leaf, t_i, y_i, yp_i, inp_i) in enumerate(
            zip(leaves, ts, ys, yps, inputs, strict=True)
        ):
            if i >= n:
                break
            is_last_segment = seg_idx == n_segments - 1
            if is_last_segment:
                j = n
            else:
                j = i + np.searchsorted(t[i:], t_i[-1], side="right")
            if j <= i:
                continue
            # Copy: the caller may mutate t, and a shared leaf holds its arguments
            query = np.array(t[i:j])
            if t_i.size < 2:
                # A single knot: hold its state constant and evaluate directly
                y_query = np.repeat(y_i, len(query), axis=1)
                cols.append(np.asarray(leaf.eval_trajectory(query, y_query, inp_i)))
            else:
                cols.append(
                    np.asarray(
                        leaf.eval_trajectory_hermite(query, t_i, y_i, yp_i, inp_i)
                    )
                )
            i = j
        return np.concatenate(cols, axis=1).reshape(variable._shape(t), order="F")

    def sensitivities(self, variable):
        segments = list(
            zip(
                self._leaves,
                variable.all_ts,
                variable.all_ys,
                variable.all_inputs,
                strict=True,
            )
        )
        sensitivity_names = variable.sensitivity_names
        sensitivities = native_sensitivities(
            segments, variable.all_solution_sensitivities["all"], sensitivity_names
        )
        if variable.time_integral is not None:
            # The value sums over every segment at once, so its sensitivity must too
            sensitivities = native_postfix_sensitivities(
                variable.time_integral,
                variable._name,
                variable.t_pts,
                self.observe_raw(variable),
                sensitivities,
                variable.all_inputs[0],
                sensitivity_names,
                self._post_sum_leaf,
            )
        return pack_sensitivity_dict(sensitivities, sensitivity_names)


def _row_spans(function):
    """``{name: slice}`` over the rows of a multi-output compiled tape."""
    spans = {}
    offset = 0
    for name, length in zip(function.output_names, function.output_lens, strict=True):
        spans[name] = slice(offset, offset + length)
        offset += length
    return spans


def _same_argument(previous, current):
    """Whether two call arguments select the same evaluation; unknowns differ."""
    if previous is current:
        return True
    if isinstance(previous, np.ndarray) and isinstance(current, np.ndarray):
        return previous.shape == current.shape and np.array_equal(previous, current)
    if isinstance(previous, dict) and isinstance(current, dict):
        return previous.keys() == current.keys() and all(
            _same_argument(previous[key], current[key]) for key in previous
        )
    if isinstance(previous, list | tuple) and isinstance(current, list | tuple):
        return len(previous) == len(current) and all(
            _same_argument(was, now) for was, now in zip(previous, current, strict=True)
        )
    if np.isscalar(previous) and np.isscalar(current):
        return previous == current
    return False


class SharedOutputs:
    """One segment's tape over several variables, evaluated once for all of them.

    A result is held until every leaf has read its rows. Held arguments must be
    arrays PyBaMM owns and never mutates; copy a caller's array first.
    """

    def __init__(self, function):
        self.function = function
        self._spans = _row_spans(function)
        self._call = None
        self._result = None
        self._unread = 0

    def __getstate__(self):
        """Pickle the tape, never a held result."""
        return {"function": self.function}

    def __setstate__(self, state):
        self.__init__(state["function"])

    def span(self, name):
        """The rows ``name`` owns in this tape's output."""
        return self._spans[name]

    def _hold(self, call, compute):
        """``compute()``'s result, reused while an identical call is still held."""
        if self._call is None or not _same_argument(self._call, call):
            self._result = compute()
            self._call = call
            self._unread = len(self._spans)
        result = self._result
        self._unread -= 1
        if self._unread <= 0:
            self._call = None
            self._result = None
        return result

    def evaluate(self, method, args):
        """``method(*args)`` on the shared tape, reusing an identical held call."""
        return self._hold(
            (method, args),
            lambda: np.asarray(getattr(self.function, method)(*args)),
        )

    def chain_rule(self, ts, ys, inputs, dy_dp, sens_names):
        """The whole set's ``dvar/dp`` for one sub-solution, swept once.

        Shape ``(n_t * function.output_len, n_p)``, time-outer / output-inner.
        """
        args = (ts, ys, inputs, dy_dp, sens_names)
        return self._hold(
            ("chain_rule", args), lambda: _sweep_sensitivities(self.function, *args)
        )


class SharedOutputLeaf:
    """One variable's rows of a :class:`SharedOutputs` tape, used as its own tape."""

    def __init__(self, shared, name):
        self._shared = shared
        self._name = name
        self._rows = shared.span(name)

    @property
    def output_len(self):
        return self._rows.stop - self._rows.start

    @property
    def input_names(self):
        return self._shared.function.input_names

    @property
    def n_inputs(self):
        return self._shared.function.n_inputs

    def eval_trajectory(self, ts, ys, inputs):
        return self._shared.evaluate("eval_trajectory", (ts, ys, inputs))[self._rows]

    def eval_trajectory_hermite(self, t_query, ts, ys, yps, inputs):
        return self._shared.evaluate(
            "eval_trajectory_hermite", (t_query, ts, ys, yps, inputs)
        )[self._rows]

    def chain_rule(self, ts, ys, inputs, dy_dp, sens_names):
        """This variable's rows of the set's chain-rule sweep."""
        full = self._shared.chain_rule(ts, ys, inputs, dy_dp, sens_names)
        n_p = len(sens_names)
        rows = full.reshape(len(ts), -1, n_p)[:, self._rows, :]
        return rows.reshape(len(ts) * self.output_len, n_p)


def as_observer(leaves: VariableObserver | list) -> VariableObserver:
    """Coerce leaves to an observer.

    Parameters
    ----------
    leaves : :class:`VariableObserver` or list of :class:`casadi.Function`
        An observer, returned unchanged, or one CasADi function per
        sub-solution, wrapped in a :class:`CasadiObserver`.

    Returns
    -------
    :class:`VariableObserver`
    """
    if isinstance(leaves, VariableObserver):
        return leaves
    return CasadiObserver(leaves)


def check_variable_in_solve(
    solution: pybamm.Solution, name: str, var_pybamm: pybamm.Symbol
) -> None:
    """Reject a state-dependent variable an outputs-only solve did not store.

    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        The solution the variable is read from.
    name : str
        The variable's name.
    var_pybamm : :class:`pybamm.Symbol`
        The variable's discretised expression.

    Raises
    ------
    KeyError
        If the solve returned variables only and ``var_pybamm`` reads states.
    """
    if not solution.variables_returned:
        return
    if var_pybamm.has_symbol_of_classes(
        pybamm.expression_tree.state_vector.StateVector
    ):
        raise KeyError(
            f"Cannot process variable '{name}' as it was not part of the "
            "solve. Please re-run the solve with `output_variables` set to "
            "include this variable."
        )


def native_sensitivities(
    segments, dy_dp_segments, sensitivity_names
) -> npt.NDArray[np.float64]:
    """Forward sensitivities ``(N, n_p)`` of a natively observed variable.

    ``segments`` holds one ``(leaf, ts, ys, inputs)`` per sub-solution, stacked in
    solve order.
    """
    return np.vstack(
        [
            chain_rule_sensitivities(leaf, ts, ys, inputs, dy_dp, sensitivity_names)
            for (leaf, ts, ys, inputs), dy_dp in zip(
                segments, dy_dp_segments, strict=True
            )
        ]
    )


def native_postfix_sensitivities(
    time_integral,
    var_name,
    t_pts,
    entries,
    sensitivities,
    inputs,
    sensitivity_names,
    post_sum_leaf,
) -> npt.NDArray[np.float64]:
    """Sensitivities ``(n_integral, n_p)`` of a time integral's value, natively."""
    time_integral.check_discrete_times(var_name, t_pts)

    # The initial condition is a constant, so it adds nothing to a sensitivity
    integral_sensitivities = time_integral.sum_over_time(sensitivities, t_pts)
    if time_integral.post_sum_node is None:
        return integral_sensitivities

    integral = np.ascontiguousarray(
        time_integral.postfix_sum(entries, t_pts), dtype=np.float64
    ).ravel()
    p_stacked = post_sum_leaf.pack(inputs)
    dpost_dy = post_sum_leaf.jacobian("y")(0.0, integral, p_stacked).toarray()
    dpost_dp_all = post_sum_leaf.jacobian("p")(0.0, integral, p_stacked).toarray()
    input_names = list(post_sum_leaf.input_names)
    dpost_dp = np.zeros((dpost_dy.shape[0], len(sensitivity_names)))
    for k, name in enumerate(sensitivity_names):
        if name in input_names:
            dpost_dp[:, k] = dpost_dp_all[:, input_names.index(name)]
    return dpost_dy @ integral_sensitivities + dpost_dp


def chain_rule_sensitivities(cf, ts, ys, inputs, dy_dp, sens_names):
    """Variable sensitivities ``dvar/dp`` for one sub-solution via jvp_trajectory.

    ``dy_dp`` is ``(n_t * n_states, n_p)`` and the result ``(n_t * output_len, n_p)``,
    both time-outer, with columns ordered as ``sens_names``.
    """
    if isinstance(cf, SharedOutputLeaf):
        return cf.chain_rule(ts, ys, inputs, dy_dp, sens_names)
    return _sweep_sensitivities(cf, ts, ys, inputs, dy_dp, sens_names)


def _sweep_sensitivities(cf, ts, ys, inputs, dy_dp, sens_names):
    """One forward sweep per sensitivity parameter over ``cf``'s whole output."""
    n_states, n_t = ys.shape
    out_len = cf.output_len
    input_names = cf.input_names
    sensitivities = np.empty((n_t * out_len, len(sens_names)))
    for k, name in enumerate(sens_names):
        vy_k = np.ascontiguousarray(dy_dp[:, k].reshape(n_t, n_states).T)
        if name in input_names:
            vp = np.zeros(cf.n_inputs)
            vp[input_names.index(name)] = 1.0
            col = cf.jvp_trajectory(ts, ys, inputs, vy_k, vp=vp)
        else:
            # variable has no direct dependence on this parameter; dvar_dp == 0
            col = cf.jvp_trajectory(ts, ys, inputs, vy_k)
        # (out_len, n_t) -> time-outer / output-inner flat column
        sensitivities[:, k] = np.asarray(col).T.reshape(-1)
    return sensitivities


def pack_sensitivity_dict(
    sensitivity_matrix: npt.NDArray[np.float64], sensitivity_names: Sequence[str]
) -> dict[str, npt.NDArray[np.float64]]:
    """Split a sensitivity matrix into its per-parameter vectors.

    Parameters
    ----------
    sensitivity_matrix : numpy.ndarray
        Sensitivities of shape ``(N, n_p)``, one column per parameter.
    sensitivity_names : list of str
        Parameter names, in column order.

    Returns
    -------
    dict
        The whole matrix under ``"all"``, plus one flat ``(N,)`` vector per
        parameter, keyed by its name.
    """
    sensitivities = {"all": sensitivity_matrix}
    for i, name in enumerate(sensitivity_names):
        sensitivities[name] = sensitivity_matrix[:, i : i + 1].reshape(-1)
    return sensitivities


def _is_f_contiguous(all_ys):
    """Whether every array in ``all_ys`` is Fortran-contiguous.

    Parameters
    ----------
    all_ys : list of numpy.ndarray
        Per-sub-solution state arrays.

    Returns
    -------
    bool
    """

    return all(isinstance(y, np.ndarray) and y.data.f_contiguous for y in all_ys)


def _find_ts_indices(starts, ends, t):
    """Segments containing at least one of ``t``.

    Parameters
    ----------
    starts, ends : list of float
        First and last time of each segment, successively increasing.
    t : array-like
        Sorted query times.

    Returns
    -------
    list of int
        Positions in ``starts``/``ends``, plus the last segment when ``t``
        extends past it.
    """

    indices = []

    t_min, t_max = t[0], t[-1]

    # Binary search for the range of segments where t_min and t_max could lie
    low_idx = bisect.bisect_left(ends, t_min)
    high_idx = bisect.bisect_right(starts, t_max)

    for idx in range(low_idx, high_idx):
        # Binary search within `t` to check if any value falls within the segment
        i = bisect.bisect_left(t, starts[idx])
        if i < len(t) and t[i] <= ends[idx]:
            indices.append(idx)

    # Past the last segment, extrapolate from it
    if (t_max > ends[-1]) and (len(indices) == 0 or indices[-1] != len(ends) - 1):
        indices.append(len(ends) - 1)

    return indices
