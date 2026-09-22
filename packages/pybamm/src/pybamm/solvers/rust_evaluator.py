"""CasADi-convention ``(t, y, stacked_inputs)`` callables over Rust tapes."""

import numpy as np


def _as_1d(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()


class RustEvaluator:
    """Role-specific view over a per-group ``CompiledFunction``.

    ``role`` is ``func``, ``jac`` (scipy CSC), ``jac_action`` or ``jacp`` (one
    ``df/dp_i`` column per ``sens_indices`` entry, as CasADi's ``jacp``).
    """

    def __init__(self, cf, role, sens_indices=None):
        self._cf = cf
        self._role = role
        self._sens_indices = sens_indices
        self._jac = None  # derived lazily on first jac/jacp call

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jac"] = None
        return state

    def _jacobian(self, wrt):
        if self._jac is None:
            self._jac = self._cf.jacobian(wrt)
        return self._jac

    def __call__(self, t, y, inputs, v=None):
        t, y, p = float(t), _as_1d(y), _as_1d(inputs)
        if self._role == "func":
            return np.asarray(self._cf(t, y, p)).reshape(-1, 1)
        if self._role == "jac":
            return self._jacobian("y")(t, y, p)
        if self._role == "jac_action":
            return np.asarray(self._cf.jvp(t, y, p, _as_1d(v))).reshape(-1, 1)
        dense = self._jacobian("p")(t, y, p).toarray()
        return tuple(dense[:, i].reshape(-1, 1) for i in self._sens_indices)
