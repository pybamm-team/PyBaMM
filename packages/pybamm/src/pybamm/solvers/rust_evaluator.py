"""CasADi-convention ``(t, y, stacked_inputs)`` callables over Rust tapes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

import pybamm

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix

    from pybamm.rust import CompiledFunction, CompiledJacobian


def _as_1d(arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()


class RustEvaluator:
    """Role-specific view over a per-group ``CompiledFunction``.

    Parameters
    ----------
    cf : :class:`pybamm.rust.CompiledFunction`
        The compiled function to evaluate.
    role : str
        ``"func"``, ``"jac"`` (scipy CSC), ``"jac_action"`` or ``"jacp"`` (one
        ``df/dp_i`` column per ``sens_indices`` entry, as CasADi's ``jacp``).
    sens_indices : list of int, optional
        Graph input indices of the sensitivity parameters, read by ``"jacp"``.
    """

    def __init__(
        self,
        cf: CompiledFunction,
        role: Literal["func", "jac", "jac_action", "jacp"],
        sens_indices: list[int] | None = None,
    ) -> None:
        self._cf = cf
        self._role = role
        self._sens_indices = sens_indices
        # Derived lazily on the first jac/jacp call
        self._jac: CompiledJacobian | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_jac"] = None
        return state

    def _jacobian(self, wrt: Literal["y", "p"]) -> CompiledJacobian:
        if self._jac is None:
            self._jac = self._cf.jacobian(wrt)
        return self._jac

    def __call__(
        self,
        t: float,
        y: npt.ArrayLike,
        inputs: npt.ArrayLike,
        v: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64] | csc_matrix | tuple[npt.NDArray[np.float64], ...]:
        """Evaluate this evaluator's role at ``(t, y, inputs)``.

        Parameters
        ----------
        t : float
            The time.
        y : array-like
            The state vector.
        inputs : array-like
            The stacked input values.
        v : array-like, optional
            The vector a ``"jac_action"`` multiplies.

        Returns
        -------
        numpy.ndarray, scipy.sparse.csc_matrix or tuple of numpy.ndarray
            A column for ``"func"`` and ``"jac_action"``, the sparse Jacobian for
            ``"jac"``, and one column per sensitivity parameter for ``"jacp"``.

        Raises
        ------
        pybamm.SolverError
            If a ``"jac_action"`` evaluator gets no ``v``, or a ``"jacp"`` evaluator
            has no ``sens_indices``.
        """
        t, y, p = float(t), _as_1d(y), _as_1d(inputs)
        if self._role == "func":
            return np.asarray(self._cf(t, y, p)).reshape(-1, 1)
        if self._role == "jac":
            return self._jacobian("y")(t, y, p)
        if self._role == "jac_action":
            if v is None:
                raise pybamm.SolverError("A 'jac_action' evaluator needs the vector v")
            return np.asarray(self._cf.jvp(t, y, p, _as_1d(v))).reshape(-1, 1)
        if self._sens_indices is None:
            raise pybamm.SolverError("A 'jacp' evaluator needs sens_indices")
        dense = self._jacobian("p")(t, y, p).toarray()
        return tuple(dense[:, i].reshape(-1, 1) for i in self._sens_indices)
