"""Callable wrappers around Rust prep-artifacts for ``BaseSolver.process``.

Rust evaluators follow the casadi calling convention (positional
``(t, y, stacked_inputs)``, all inputs stacked into one flat vector) but take and
return plain numpy / scipy-sparse objects, never ``casadi.DM``. Every role wraps one
``CompiledFunction``; the ``CompiledJacobian`` backing the ``jac``/``jacp`` roles is
derived lazily so groups whose jacobian is never evaluated (idaklu/diffsol, which
integrate via ``CompiledModel``) pay nothing for it. See the bundle-accessor API in
``packages/pybamm-rust/pybamm-python/src/{function,jacobian}.rs``.
"""

import numpy as np


def _as_1d(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()


class RustEvaluator:
    """Role-specific view over a per-group ``CompiledFunction``.

    Roles:
      - ``func``       -> column vector ``f(t, y, p)``
      - ``jac``        -> ``df/dy`` as scipy CSC          (lazy ``cf.jacobian("y")``)
      - ``jac_action`` -> ``(df/dy) @ v`` column          (``cf.jvp``)
      - ``jacp``       -> tuple of ``df/dp_i`` columns, one per sensitivity
        parameter (lazy ``cf.jacobian("p")`` sliced to ``sens_indices``), matching
        casadi's multi-output ``jacp`` convention

    Parameters
    ----------
    cf : :class:`pybamm.rust.CompiledFunction`
        The compiled function this evaluator views.
    role : str
        One of the roles above; it fixes what calling the evaluator returns.
    sens_indices : list[int], optional
        Positions in ``cf``'s parameter vector to keep for the ``jacp`` role,
        in the order the solver expects its sensitivity columns.
    """

    def __init__(self, cf, role, sens_indices=None):
        self._cf = cf
        self._role = role
        self._sens_indices = sens_indices
        self._jac = None  # CompiledJacobian, derived on first jac/jacp call

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jac"] = None  # derived cache; rebuilt lazily after unpickle
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
            return self._jacobian("y")(t, y, p)  # scipy CSC
        if self._role == "jac_action":
            return np.asarray(self._cf.jvp(t, y, p, _as_1d(v))).reshape(-1, 1)
        # jacp: slice df/dp columns to the requested sensitivity params
        dense = self._jacobian("p")(t, y, p).toarray()
        return tuple(dense[:, i].reshape(-1, 1) for i in self._sens_indices)
