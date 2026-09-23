"""Per-variable observation strategies for :class:`pybamm.ProcessedVariable`.

A :class:`VariableObserver` owns one variable's per-sub-solution leaves and
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
    ``all_solution_sensitivities``, ``data``, ``_name`` and ``_shape``.
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
        # Keyed to the time arrays it was built from: an observer can be
        # shared by variables of different solutions.
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

            # Set up symbolic variables
            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in sensitivity_inputs.items()
            }

            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Non-target inputs can still appear in the tree (e.g. from experiment
            # steps), so they stay concrete while the targets go symbolic.
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
                    variable._name, variable.data, ts, inputs, S_var
                )

            all_S_var.append(S_var)

        return pack_sensitivity_dict(np.vstack(all_S_var), sensitivity_names)


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

    # Get the minimum and maximum values of the target values `t`
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
