"""Per-variable observation strategies for :class:`pybamm.ProcessedVariable`.

A :class:`VariableObserver` owns one variable's per-sub-solution leaves and
knows how to evaluate them on the solution grid, off-grid via cubic Hermite,
and through the forward chain rule for sensitivities. ``ProcessedVariable``
holds exactly one, chosen when it is built, so nothing downstream re-decides
which backend is in play.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod

import casadi
import numpy as np
from pybammsolvers import idaklu

import pybamm


class SegmentSelector:
    """The single rule for which sub-solutions cover a set of query times.

    A variable's ``all_ts`` is frozen for its lifetime, so the non-empty
    segments and their end times are found once here rather than on every
    observe call.

    Parameters
    ----------
    all_ts : list[numpy.ndarray]
        Per-sub-solution time arrays, successively increasing.
    """

    def __init__(self, all_ts):
        self.indices = np.where([ti.size > 0 for ti in all_ts])[0]
        self._starts = [all_ts[idx][0] for idx in self.indices]
        self._ends = [all_ts[idx][-1] for idx in self.indices]

    def select(self, t, full_range):
        """Indices into ``all_ts`` covering ``t``, ascending.

        Empty segments are always dropped; when ``full_range`` is False only
        the segments whose span contains at least one of ``t`` are kept.
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
    ``all_solution_sensitivities`` and ``_shape``.
    """

    #: Derived caches: built on demand, never pickled (see __getstate__).
    _selector = None
    _serialised = None

    def __getstate__(self):
        """Pickle without the derived caches, which a Solution should not carry."""
        state = self.__dict__.copy()
        for key in ("_selector", "_serialised"):
            state.pop(key, None)
        return state

    @property
    @abstractmethod
    def leaves(self) -> list:
        """The variable's per-sub-solution evaluable leaves, in solve order."""

    def segments(self, variable, t, full_range):
        """Indices of ``variable``'s sub-solutions covering ``t``."""
        if self._selector is None:
            self._selector = SegmentSelector(variable.all_ts)
        return self._selector.select(t, full_range)

    @abstractmethod
    def observe_raw(self, variable):
        """Evaluate on the solution's own time points, shaped by ``_shape``."""

    @abstractmethod
    def observe_hermite(self, variable, t):
        """Evaluate at arbitrary sorted times ``t``, cubic-Hermite in state."""

    @abstractmethod
    def sensitivities(self, variable) -> dict:
        """``{"all": (N, n_p), param: (N,)}`` forward sensitivities of the variable."""


class CasadiObserver(VariableObserver):
    """Observation through serialised CasADi functions and the IDAKLU kernels.

    Parameters
    ----------
    leaves : list of :class:`casadi.Function`
        One function per sub-solution, evaluating ``(t, y, p_stacked)``.
    """

    def __init__(self, leaves):
        self._leaves = leaves

    @property
    def leaves(self):
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

            # Set up symbolic variables
            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in sensitivity_inputs.items()
            }

            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Symbolic for sensitivity targets, concrete for the rest. Non-target
            # inputs may still appear in the expression tree (e.g. from
            # experiment steps) so they must be present for casadi conversion.
            inputs_for_casadi = {**inputs, **p_casadi}

            var_casadi = base_variable.to_casadi(
                t_casadi, y_casadi, inputs=inputs_for_casadi
            )
            dvar_dy = casadi.jacobian(var_casadi, y_casadi)
            dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

            # Convert to functions and evaluate index-by-index
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

            # Compute sensitivity
            S_var = dvar_dy_eval @ dy_dp + dvar_dp_eval

            if variable.time_integral is not None:
                S_var = variable.time_integral.postfix_sensitivities(
                    variable.name, variable.data, ts, inputs, S_var
                )

            all_S_var.append(S_var)

        return pack_sensitivity_dict(np.vstack(all_S_var), sensitivity_names)


class NativeObserver(VariableObserver):
    """Observation through compiled Rust tapes lowered into the retained graph.

    Parameters
    ----------
    leaves : list of :class:`pybamm.rust.CompiledFunction`
        One compiled tape per sub-solution, evaluating ``(t, y, inputs)``.
    backend : :class:`pybamm.solvers.observation.NativeObservation`
        Owner of the retained graph and the compile cache; consulted for the
        time-integral post-sum tape.
    placeholder_states : list[int] or None
        Per-sub-solution state count for outputs-only solves, which store no
        states and need a shaped zero trajectory. ``None`` when states are real.
    """

    def __init__(self, leaves, backend, placeholder_states=None):
        self._leaves = leaves
        self._backend = backend
        self._placeholder_states = placeholder_states

    @property
    def leaves(self):
        return self._leaves

    def _setup(self, variable, t, full_range):
        """Per-segment ``(ts, ys, yps, inputs, leaves)`` as plain numpy and dicts.

        Inputs stay dict-shaped: the compiled tape packs by name, so there are
        no stacking-order concerns.
        """
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
        # eval_trajectory returns (output_len, n_t) F-contiguous, so the flat
        # spatial index varies fastest and reshape must use order="F".
        t = variable.t_pts
        ts, ys, _, inputs, leaves = self._setup(variable, t, full_range=True)
        cols = [
            np.asarray(leaf.eval_trajectory(t_i, y_i, inp_i))
            for leaf, t_i, y_i, inp_i in zip(leaves, ts, ys, inputs, strict=True)
        ]
        return np.concatenate(cols, axis=1).reshape(variable._shape(t), order="F")

    def observe_hermite(self, variable, t):
        # Route each query time to its segment (mirrors observe.cpp's sequential
        # <=/> knot-window scan), Hermite-reconstruct, then concatenate.
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
            query = t[i:j]
            if t_i.size < 2:
                # No interval to Hermite-interpolate within: fall back to a
                # direct eval, holding the single known state constant.
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
        return native_sensitivities(
            segments,
            variable.all_solution_sensitivities["all"],
            variable.sensitivity_names,
            time_integral=variable.time_integral,
            postfix=lambda inner, sens_names: self._backend.postfix_sensitivities(
                variable.t_pts,
                variable.name,
                variable.time_integral,
                variable.entries,
                inner,
                variable.all_inputs[0],
                sens_names,
            ),
        )


def as_observer(leaves) -> VariableObserver:
    """Coerce a :class:`ProcessedVariable` leaf argument to an observer.

    A bare list of :class:`casadi.Function` — the historical form, and what the
    default backend passes — becomes a :class:`CasadiObserver`.
    """
    if isinstance(leaves, VariableObserver):
        return leaves
    return CasadiObserver(leaves)


def check_variable_in_solve(solution, name, var_pybamm) -> None:
    """Reject a state-dependent variable an outputs-only solve did not store.

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


def native_sensitivities(segments, dy_dp_segments, sens_names, time_integral, postfix):
    """Forward sensitivities of a natively-observed variable, packed for reading.

    Parameters
    ----------
    segments : list[tuple]
        One ``(compiled leaf, ts, ys, inputs)`` per sub-solution, in solve order.
    dy_dp_segments : list[numpy.ndarray]
        The matching state sensitivities, one block per sub-solution.
    sens_names : list[str]
        Sensitivity-parameter names; the column order throughout.
    time_integral : pybamm.ProcessedVariableTimeIntegral or None
        Set when the variable is time-integrated, so the chain rule runs on the
        integrand and ``postfix`` finishes it.
    postfix : callable
        ``(inner_sensitivities, sens_names) -> dvar/dp``, applying the post-sum
        chain rule at the postfix value. Only called for a time integral.

    Returns
    -------
    dict
        ``{"all": (N, n_p), param: (N,)}``, as :meth:`ProcessedVariable.sensitivities`.
    """
    inner = np.vstack(
        [
            chain_rule_sensitivities(leaf, ts, ys, inputs, dy_dp, sens_names)
            for (leaf, ts, ys, inputs), dy_dp in zip(
                segments, dy_dp_segments, strict=True
            )
        ]
    )
    if time_integral is None:
        return pack_sensitivity_dict(inner, sens_names)
    return pack_sensitivity_dict(postfix(inner, sens_names), sens_names)


def chain_rule_sensitivities(cf, ts, ys, inputs, dy_dp, sens_names):
    """Variable sensitivities ``dvar/dp`` for one sub-solution via jvp_trajectory.

    Computes ``S_var[:, k] = dvar_dy(t)·yS_k(t) + dvar_dp(t)·e_k`` for every
    sensitivity parameter, one Rust forward sweep per parameter.

    Parameters
    ----------
    cf : pybamm.rust.CompiledFunction
        Compiled observed-variable function.
    ts : numpy.ndarray
        Sub-solution times, shape ``(n_t,)``.
    ys : numpy.ndarray
        State trajectory, shape ``(n_states, n_t)``.
    inputs : dict or numpy.ndarray
        Input-parameter values for this sub-solution.
    dy_dp : numpy.ndarray
        State sensitivities, shape ``(n_t * n_states, n_p)``, time-outer /
        state-inner, columns ordered as ``sens_names``.
    sens_names : list[str]
        Sensitivity-parameter names; column order of ``dy_dp``.

    Returns
    -------
    numpy.ndarray
        Variable sensitivities, shape ``(n_t * output_len, n_p)``, time-outer /
        output-inner — the layout the CasADi forward path produces.
    """
    n_states, n_t = ys.shape
    out_len = cf.output_len
    input_names = cf.input_names
    sensitivities = np.empty((n_t * out_len, len(sens_names)))
    for k, name in enumerate(sens_names):
        # yS for this parameter, reshaped (n_t, n_states) then to (n_states, n_t)
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


def pack_sensitivity_dict(S_var, sens_names):
    """Pack a sensitivity matrix ``(N, n_p)`` into the ``{"all", per-param}`` dict.

    Matches :meth:`ProcessedVariable.sensitivities`: the full block under
    ``"all"`` plus one flat ``(N,)`` vector per parameter in ``sens_names`` order.
    """
    sensitivities = {"all": S_var}
    for i, name in enumerate(sens_names):
        sensitivities[name] = S_var[:, i : i + 1].reshape(-1)
    return sensitivities


def _is_f_contiguous(all_ys):
    """
    Check if all the ys are f-contiguous in memory

    Args:
        all_ys (list of np.ndarray): list of all ys

    Returns:
        bool: True if all ys are f-contiguous
    """

    return all(isinstance(y, np.ndarray) and y.data.f_contiguous for y in all_ys)


def _find_ts_indices(starts, ends, t):
    """
    Parameters:
    - starts, ends: First and last time of each segment, successively increasing.
    - t: A sorted list or array of values to find within the segments.

    Returns:
    - indices: Positions in `starts`/`ends` whose segment contains a value of `t`.
    """

    indices = []

    # Get the minimum and maximum values of the target values `t`
    t_min, t_max = t[0], t[-1]

    # Step 1: Use binary search to find the range of segments where t_min and t_max could lie
    low_idx = bisect.bisect_left(ends, t_min)
    high_idx = bisect.bisect_right(starts, t_max)

    # Step 2: Iterate over the identified range
    for idx in range(low_idx, high_idx):
        # Binary search within `t` to check if any value falls within the segment
        i = bisect.bisect_left(t, starts[idx])
        if i < len(t) and t[i] <= ends[idx]:
            indices.append(idx)

    # extrapolating
    if (t_max > ends[-1]) and (len(indices) == 0 or indices[-1] != len(ends) - 1):
        indices.append(len(ends) - 1)

    return indices
