"""Per-variable observation strategies for :class:`pybamm.ProcessedVariable`.

A :class:`VariableObserver` holds one variable's per-sub-solution leaves and
evaluates them on the solution grid, off-grid via cubic Hermite, and through
the forward chain rule for sensitivities.
"""

from __future__ import annotations

import bisect
import numbers
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

    #: Gaps between segments of at most this many units in the last place of
    #: their times are round-off, like those PyBaMM leaves at discontinuities
    roundoff_gap_ulps = 8

    def __init__(self, all_ts: Sequence[npt.NDArray[np.float64]]):
        self.indices = np.where([np.size(ti) > 0 for ti in all_ts])[0]
        self._starts = [all_ts[idx][0] for idx in self.indices]
        self._ends = [all_ts[idx][-1] for idx in self.indices]

    def resolve_gaps(
        self, t: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_] | None]:
        """Place times that fall between segments.

        A time in a round-off gap between two segments, such as the one PyBaMM
        leaves around a discontinuity, takes the end of the earlier segment (the
        left limit). A time in a longer gap, such as one left by cycles that
        ``save_at_cycles`` dropped, has no value.

        Parameters
        ----------
        t : numpy.ndarray
            Query times, of any shape.

        Returns
        -------
        t : numpy.ndarray
            The times, with those in round-off gaps moved.
        in_gap : numpy.ndarray of bool or None
            Which times lie in a longer gap, or None if none do.
        """
        if len(self._starts) < 2:
            return t, None
        t = np.asarray(t, dtype=float)
        starts = np.asarray(self._starts, dtype=float)
        ends = np.asarray(self._ends, dtype=float)
        # The segment that starts last at or before each time
        segment = np.searchsorted(starts, t, side="right") - 1
        between = (segment >= 0) & (segment < len(starts) - 1)
        segment = np.where(between, segment, 0)
        between &= t > ends[segment]
        if not between.any():
            return t, None
        next_start = starts[segment + 1]
        roundoff = between & (
            next_start - ends[segment]
            <= self.roundoff_gap_ulps * np.spacing(np.abs(next_start))
        )
        t = np.where(roundoff, ends[segment], t)
        in_gap = between & ~roundoff
        return t, (in_gap if in_gap.any() else None)

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


def fill_gaps(
    values: npt.NDArray[np.float64],
    in_gap: npt.NDArray[np.bool_] | None,
    fill_value: float | str,
) -> npt.NDArray[np.float64]:
    """Set the values at times in a gap between sub-solutions, which have none.

    Parameters
    ----------
    values : numpy.ndarray
        Interpolated values, time last (or at a single time).
    in_gap : numpy.ndarray of bool or None
        Which times lie in a gap (see :meth:`SegmentSelector.resolve_gaps`),
        or None if none do.
    fill_value : float or str
        The value to set; NaN if it is not a number (e.g. ``"extrapolate"``).

    Returns
    -------
    numpy.ndarray
        ``values``, modified in place.
    """
    if in_gap is None:
        return values
    if not isinstance(fill_value, numbers.Number):
        fill_value = np.nan
    if in_gap.ndim == 0:
        values[...] = fill_value
    else:
        values[..., in_gap] = fill_value
    return values


class VariableObserver(ABC):
    """How one variable's leaves are evaluated over a solution's segments.

    Implementations read only these attributes of the ``variable`` handed to
    them: ``all_ts``, ``all_ys``, ``all_yps``, ``all_inputs``,
    ``all_inputs_stacked``, ``t_pts``, ``hermite_interpolation``,
    ``time_integral``, ``base_variables``, ``sensitivity_names``,
    ``all_solution_sensitivities``, ``_name`` and ``_shape``.
    """

    #: Caches, built on demand or shared with a model, never pickled (see
    #: __getstate__).
    _selector = None
    _selector_ts = None
    _serialised = None
    _chain_rules = None

    def __getstate__(self):
        """The observer's state without its derived caches, rebuilt on demand."""
        state = self.__dict__.copy()
        for key in ("_selector", "_selector_ts", "_serialised", "_chain_rules"):
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
        return self._segment_selector(variable).select(t, full_range)

    def resolve_gaps(
        self, variable: pybamm.ProcessedVariable, t: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_] | None]:
        """Place times that fall between ``variable``'s sub-solutions.

        See :meth:`SegmentSelector.resolve_gaps`.
        """
        return self._segment_selector(variable).resolve_gaps(t)

    def _segment_selector(self, variable):
        # An observer can be shared across solutions, so key on their time arrays
        if self._selector_ts is not variable.all_ts:
            self._selector = SegmentSelector(variable.all_ts)
            self._selector_ts = variable.all_ts
        return self._selector

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
    chain_rules : list of dict, optional
        One store per sub-solution, such as its model's
        :attr:`~pybamm.solvers.observation.ObserverCache.chain_rules`, that keeps
        the chain-rule functions built for its sensitivities for later reads. By
        default they are kept for one read.
    """

    def __init__(self, leaves: list, chain_rules: list[dict] | None = None):
        self._leaves = leaves
        self._chain_rules = chain_rules

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
        all_dy_dp = variable.all_solution_sensitivities.get("all", [])
        if len(all_dy_dp) != len(variable.all_ts):
            raise pybamm.SolverError(
                f"Cannot compute the sensitivities of {variable._name!r}: the "
                f"solution has state sensitivities for {len(all_dy_dp)} of its "
                f"{len(variable.all_ts)} segments."
            )
        # One chain-rule function per distinct leaf and input layout, as
        # experiment steps and repeated solves that share a model repeat them
        stores = self._chain_rules
        if stores is None or len(stores) != len(variable.all_ts):
            stores = [{}] * len(variable.all_ts)
        all_S_var = []
        for ts, ys, inputs, base_variable, dy_dp, store in zip(
            variable.all_ts,
            variable.all_ys,
            variable.all_inputs,
            variable.base_variables,
            all_dy_dp,
            stores,
            strict=True,
        ):
            missing = [name for name in sensitivity_names if name not in inputs]
            if missing:
                raise pybamm.SolverError(
                    f"Cannot compute the sensitivities of {variable._name!r}: a "
                    f"segment of the solution has no input values for {missing}."
                )
            sensitivity_values = [
                np.asarray(inputs[name], dtype=float).reshape(-1)
                for name in sensitivity_names
            ]
            other_names = [name for name in inputs if name not in sensitivity_names]
            other_values = [
                np.asarray(inputs[name], dtype=float).reshape(-1)
                for name in other_names
            ]
            n_times, n_states = len(ts), ys.shape[0]
            n_params = sum(value.size for value in sensitivity_values)
            dy_dp = np.asarray(dy_dp)
            if dy_dp.shape != (n_times * n_states, n_params):
                raise pybamm.SolverError(
                    f"Cannot compute the sensitivities of {variable._name!r}: a "
                    f"segment's state sensitivities have shape {dy_dp.shape}, but "
                    f"its {n_times} times, {n_states} states and {n_params} "
                    f"sensitivity parameters need {(n_times * n_states, n_params)}."
                )
            if n_times == 0:
                continue

            sensitivity_inputs = dict(
                zip(sensitivity_names, sensitivity_values, strict=True)
            )
            other_inputs = dict(zip(other_names, other_values, strict=True))
            key = (
                id(base_variable),
                n_states,
                tuple((name, value.size) for name, value in sensitivity_inputs.items()),
                tuple((name, value.size) for name, value in other_inputs.items()),
            )
            entry = store.get(key)
            if entry is None:
                # Holding the expression keeps its id from being reused while stored
                entry = (
                    base_variable,
                    _ChainRule(
                        base_variable, n_states, sensitivity_inputs, other_inputs
                    ),
                )
                store[key] = entry
            function = entry[1]
            all_S_var.append(
                function.evaluate(
                    ts,
                    ys,
                    np.concatenate([np.zeros(0), *sensitivity_values]),
                    np.concatenate([np.zeros(0), *other_values]),
                    dy_dp,
                )
            )

        if all_S_var:
            S_var = np.vstack(all_S_var)
        else:
            S_var = np.zeros((0, len(sensitivity_names)))
        time_integral = variable.time_integral
        if time_integral is not None:
            # The value sums over every segment at once, so its sensitivity must too
            t_pts = variable.t_pts
            indices = time_integral.checked_sample_indices(
                variable._name,
                t_pts,
                "the solution's sensitivities cannot be interpolated",
            )
            n_params = S_var.shape[1]
            # Rows are time-major: one block of the variable's entries per time
            S_var = S_var.reshape(len(t_pts), -1, n_params)[indices]
            integrand = np.moveaxis(self.observe_raw(variable), -1, 0)[indices]
            S_var = time_integral.postfix_sensitivities(
                variable._name,
                integrand,
                t_pts[indices],
                variable.all_inputs[0],
                sensitivity_names,
                S_var.reshape(-1, n_params),
            )
        return pack_sensitivity_dict(S_var, sensitivity_names)


class _ChainRule:
    """A variable's forward sensitivities over many times, by the chain rule.

    At each time, the sensitivity to parameter ``k`` is ``dvar/dy @ dy/dp_k +
    dvar/dp @ e_k``, a pair of directional derivatives. They are evaluated over
    blocks of times at once, reading the state and its sensitivities in place.
    The mapped functions are kept, so later evaluations reuse them.

    Parameters
    ----------
    base_variable : :class:`pybamm.Symbol`
        The variable's discretised expression.
    n_states : int
        The length of the state vector.
    sensitivity_inputs : dict
        The inputs to differentiate with respect to, in column order; only their
        sizes are used.
    other_inputs : dict
        The remaining inputs; only their sizes are used.
    """

    #: Most times per evaluation, a power of two; building a map costs time
    #: linear in its size, far more than evaluating it
    block_size = 64

    def __init__(
        self,
        base_variable: pybamm.Symbol,
        n_states: int,
        sensitivity_inputs: dict[str, npt.NDArray[np.float64]],
        other_inputs: dict[str, npt.NDArray[np.float64]],
    ):
        t = casadi.MX.sym("t")
        y = casadi.MX.sym("y", n_states)
        symbols = {
            name: casadi.MX.sym(name, value.size)
            for name, value in {**sensitivity_inputs, **other_inputs}.items()
        }
        p_sensitivity = casadi.vertcat(*[symbols[name] for name in sensitivity_inputs])
        p_other = casadi.vertcat(*[symbols[name] for name in other_inputs])
        dy_dp = casadi.MX.sym("dy_dp", n_states)
        direction = casadi.MX.sym("direction", p_sensitivity.shape[0])
        var = base_variable.to_casadi(t, y, inputs=symbols)
        total = casadi.densify(
            casadi.jtimes(var, y, dy_dp) + casadi.jtimes(var, p_sensitivity, direction)
        )
        function = casadi.Function(
            "chain_rule", [t, y, p_sensitivity, p_other, dy_dp, direction], [total]
        )
        try:
            function = function.expand()
        except RuntimeError as error:
            # Some leaves, like interpolants, have no SX form
            if "'eval_sx' not defined for" not in str(error):
                raise  # pragma: no cover
        self._function = function
        self._n_states = n_states
        self._n_params = p_sensitivity.shape[0]
        self.var_len = function.numel_out(0)
        self._maps = {}

    def _mapped(self, n_times: int) -> casadi.Function:
        """The function mapped over ``n_times`` times, built on first use."""
        mapped = self._maps.get(n_times)
        if mapped is None:
            # Inputs 2, 3 and 5 (the parameters and direction) are shared by all
            mapped = self._function.map("chain_rule", "serial", n_times, [2, 3, 5], [])
            if not mapped.sparsity_out(0).is_dense():
                # A map over one time drops an all-zero output's nonzeros, which
                # would leave that block of the result unwritten
                arguments = mapped.mx_in()
                mapped = casadi.Function(
                    "chain_rule", arguments, [casadi.densify(mapped(*arguments))]
                )
            self._maps[n_times] = mapped
        return mapped

    def _blocks(self, n_times: int):
        """Spans ``(start, stop)`` covering ``n_times`` times, in order.

        Full blocks come first, then halving powers of two, so evaluating any
        number of times builds at most ``log2(block_size) + 1`` maps.
        """
        start, size = 0, self.block_size
        while start < n_times:
            while start + size > n_times:
                size //= 2
            yield start, start + size
            start += size

    def evaluate(
        self,
        ts: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        p_sensitivity: npt.NDArray[np.float64],
        p_other: npt.NDArray[np.float64],
        dy_dp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The sensitivities of the variable at each of ``ts``.

        Parameters
        ----------
        ts : numpy.ndarray
            The times, of shape ``(n_t,)``.
        ys : numpy.ndarray
            The states at each time, of shape ``(n_states, n_t)``.
        p_sensitivity, p_other : numpy.ndarray
            The stacked input values.
        dy_dp : numpy.ndarray
            The state sensitivities, of shape ``(n_t * n_states, n_params)`` with
            time outer.

        Returns
        -------
        numpy.ndarray
            Of shape ``(n_t * var_len, n_params)``, time outer.
        """
        n_times, n_states, var_len = len(ts), self._n_states, self.var_len
        # Column-major, as CasADi reads them: each time's states are contiguous
        ts = np.ascontiguousarray(ts, dtype=float)
        ys = np.asfortranarray(ys, dtype=float).reshape(-1, order="F")
        p_sensitivity = np.ascontiguousarray(p_sensitivity, dtype=float)
        p_other = np.ascontiguousarray(p_other, dtype=float)
        directions = np.eye(self._n_params)
        S_var = np.empty((n_times * var_len, self._n_params), order="F")
        # Buffers are per call, so concurrent calls never share their memory
        buffers = {}
        for k in range(self._n_params):
            dy_dp_k = np.ascontiguousarray(dy_dp[:, k], dtype=float)
            S_var_k = S_var[:, k]
            for start, stop in self._blocks(n_times):
                if stop - start not in buffers:
                    buffers[stop - start] = self._mapped(stop - start).buffer()
                buffer, run = buffers[stop - start]
                buffer.set_arg(0, memoryview(ts[start:stop]))
                buffer.set_arg(1, memoryview(ys[start * n_states : stop * n_states]))
                buffer.set_arg(2, memoryview(p_sensitivity))
                buffer.set_arg(3, memoryview(p_other))
                buffer.set_arg(
                    4, memoryview(dy_dp_k[start * n_states : stop * n_states])
                )
                buffer.set_arg(5, memoryview(directions[k]))
                buffer.set_res(0, memoryview(S_var_k[start * var_len : stop * var_len]))
                run()
        return S_var


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
