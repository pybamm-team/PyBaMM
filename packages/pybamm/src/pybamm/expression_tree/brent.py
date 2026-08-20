#
# A bracketed scalar rootfind as an expression tree node
#
from __future__ import annotations

import casadi
import numpy as np

import pybamm


class BrentUnknown(pybamm.Symbol):
    """
    The scalar a :class:`Brent` solves for.

    Bound by the rootfinder, not by the model, so it is deliberately not a
    :class:`pybamm.Variable`: the checks that enumerate model states must not count it
    as one, or a ``Brent`` inside ``model.rhs`` or ``model.algebraic`` looks like an
    extra unknown with no equation.

    Two unknowns of the same name are the same unknown, as for any other symbol. Give
    each one its own name: a ``Brent`` nested inside another whose unknown shares its
    name shadows the outer binding, and CasADi rejects the oracle it builds.

    Parameters
    ----------
    name : str
        Name of the node. Must be unique among the unknowns it is nested with.
    """

    def _evaluate_for_shape(self):
        # a scalar, but shaped like every other column-vector node so that the
        # broadcasting helpers can read shape[1]
        return np.nan * np.ones((1, 1))

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        try:
            return inputs[self.name]
        except KeyError:
            raise pybamm.ModelError(
                f"'{self}' only has a value inside the Brent that solves for it"
            ) from None

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        # set by the enclosing Brent while it iterates
        return self._value

    _value = np.nan * np.ones((1, 1))


def _nodes_reading(root: pybamm.Symbol, unknown: pybamm.Symbol) -> set:
    """The symbols in ``root``'s graph whose subtree contains ``unknown``.

    Visits each node once; ``pre_order`` re-yields shared nodes once per path.
    """
    reads: set = set()
    seen: set = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            if node == unknown or any(child in reads for child in node.children):
                reads.add(node)
        elif node not in seen:
            seen.add(node)
            stack.append((node, True))
            stack.extend((child, False) for child in node.children)
    return reads


class _OracleCache(dict):
    """Conversion cache that keeps a :class:`Brent`'s own binding out of the shared one.

    Keys in ``local`` are held here and discarded with the oracle; the rest is written
    through to ``shared``, so the graph a rootfind shares with its surroundings is
    converted once.
    """

    def __init__(self, shared: dict, local: set):
        super().__init__()
        self._shared = shared
        self._local = local

    def get(self, key, default=None):
        value = super().get(key)
        return self._shared.get(key, default) if value is None else value

    def __setitem__(self, key, value):
        if key in self._local:
            super().__setitem__(key, value)
        else:
            self._shared[key] = value


class Brent(pybamm.Symbol):
    """
    Solve ``residual == 0`` for ``unknown`` within ``bounds``, by Brent's method.

    Every argument is an expression, so the bounds and anything inside ``residual``
    may be a :class:`pybamm.InputParameter` or any other symbol. Nothing is solved in
    Python: the node converts to a CasADi ``rootfinder`` using the native ``brent``
    plugin registered by ``pybammsolvers``, so the whole solve runs inside the CasADi
    graph.

    Brent needs only a sign change over the bounds, so it converges where a Newton
    iteration stalls, and the answer cannot leave them. Derivatives come from CasADi's
    implicit function theorem, exactly.

    Parameters
    ----------
    residual : :class:`pybamm.Symbol`
        The expression to drive to zero. Must contain ``unknown``. To invert ``f`` at
        a target, pass ``f - target``.
    unknown : :class:`pybamm.BrentUnknown`
        The value being solved for. Must appear in ``residual`` and nowhere else in
        the surrounding expression.
    bounds : tuple
        ``(lo, hi)``, the bracket to search. Either may be an expression.
    abstol : float, optional
        Absolute tolerance on the unknown. Unlike ``bounds``, a plain number, fixed
        when the node is built.
    max_iter : int, optional
        Iteration cap. A plain number, as ``abstol``.
    name : str, optional
        Name of the node.

    Examples
    --------
    .. code-block:: python

        # invert an open-circuit potential at a given voltage
        sto = pybamm.BrentUnknown("stoichiometry")
        node = pybamm.Brent(param.n.prim.U(sto, T) - voltage, sto, (0, 1))
    """

    def __init__(
        self,
        residual: pybamm.Symbol,
        unknown: BrentUnknown,
        bounds: tuple,
        *,
        abstol: float = 1e-14,
        max_iter: int = 100,
        name: str = "brent",
    ):
        if not isinstance(residual, pybamm.Symbol):
            raise TypeError(
                f"residual must be a pybamm.Symbol, got {type(residual).__name__}"
            )
        if not isinstance(unknown, BrentUnknown):
            raise TypeError(
                f"unknown must be a pybamm.BrentUnknown, got {type(unknown).__name__}"
            )
        if not any(node == unknown for node in residual.pre_order()):
            raise pybamm.ModelError(f"'{unknown}' does not appear in '{residual}'")
        if len(bounds) != 2:
            raise pybamm.ModelError(f"bounds must be a (lo, hi) pair, got {bounds}")
        self.abstol = abstol
        self.max_iter = max_iter
        # the unknown is a child so that it survives copying and serialisation with
        # the rest of the tree; it also appears inside the residual
        children = [
            residual,
            unknown,
            *(pybamm.convert_to_symbol(bound) for bound in bounds),
        ]
        super().__init__(name, children=children)

    def set_id(self):
        # the conversion cache is keyed on the id, so two Brents differing only in
        # tolerance must not be served each other's rootfinder
        super().set_id()
        self._id = hash((self._id, self.abstol, self.max_iter))

    @property
    def residual(self) -> pybamm.Symbol:
        return self.children[0]

    @property
    def unknown(self) -> BrentUnknown:
        return self.children[1]

    @property
    def bounds(self) -> tuple[pybamm.Symbol, pybamm.Symbol]:
        return tuple(self.children[2:])

    def create_copy(self, new_children=None, perform_simplifications=True):
        residual, unknown, lo, hi = self._children_for_copying(new_children)
        return Brent(
            residual,
            unknown,
            (lo, hi),
            abstol=self.abstol,
            max_iter=self.max_iter,
            name=self.name,
        )

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def to_json(self):
        json_dict = super().to_json()
        json_dict.update({"abstol": self.abstol, "max_iter": self.max_iter})
        return json_dict

    @classmethod
    def _from_json(cls, snippet: dict):
        residual, unknown, lo, hi = snippet["children"]
        return cls(
            residual,
            unknown,
            (lo, hi),
            abstol=snippet["abstol"],
            max_iter=snippet["max_iter"],
            name=snippet["name"],
        )

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        unknown = casadi.MX.sym(f"brent_unknown_{abs(self.id)}")
        cache = _OracleCache(
            casadi_symbols, _nodes_reading(self.residual, self.unknown)
        )
        equation = self.children[0]._to_casadi_inner(
            t, y, y_dot, {**inputs, self.unknown.name: unknown}, cache
        )
        lo, hi = (
            bound._to_casadi_inner(t, y, y_dot, inputs, casadi_symbols)
            for bound in self.bounds
        )

        # Pass only the symbols the residual actually reads. Handing the oracle the
        # whole state vector would copy it into the solve on every evaluation.
        free = [s for s in casadi.symvar(equation) if not casadi.is_equal(s, unknown)]
        # the plugin takes the bracket from inputs 1 and 2, which the residual itself
        # does not read, so they are declared here and left unused
        lo_sym, hi_sym = casadi.MX.sym("lo"), casadi.MX.sym("hi")
        oracle = casadi.Function(
            f"brent_oracle_{abs(self.id)}", [unknown, lo_sym, hi_sym, *free], [equation]
        )
        solver = casadi.rootfinder(
            f"brent_{abs(self.id)}",
            "brent",
            oracle,
            {"abstol": self.abstol, "max_iter": self.max_iter},
        )
        # the guess is required by the rootfinder interface and ignored by a bracketed
        # method, so either end of the bracket does
        return solver(lo, lo, hi, *free)

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """Solve with :func:`scipy.optimize.brentq`, for the non-CasADi path."""
        from scipy.optimize import brentq

        def scalar(value):
            return np.asarray(value, dtype=float).reshape(-1)[0]

        lo, hi = (scalar(b.evaluate(t, y, y_dot, inputs)) for b in self.bounds)
        unknown = self.unknown
        if not isinstance(unknown, BrentUnknown):
            raise TypeError(
                "NumPy evaluation needs a pybamm.BrentUnknown as the unknown, got "
                f"{type(unknown).__name__}"
            )

        def g(value):
            unknown._value = np.array([[value]])
            return scalar(self.residual.evaluate(t, y, y_dot, inputs))

        # a shape probe carries NaN through the bounds, so there is nothing to solve
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return np.nan * np.ones((1, 1))

        try:
            root = brentq(g, lo, hi, xtol=self.abstol, maxiter=self.max_iter)
        except (ValueError, RuntimeError):
            # a probe walks the tree before the parameters are in place
            return np.nan * np.ones((1, 1))
        finally:
            unknown._value = np.nan * np.ones((1, 1))
        return np.array([[root]])

    def diff(self, variable):
        raise NotImplementedError(
            "Brent has no symbolic derivative; use convert_to_format='casadi'."
        )

    def _jac(self, variable):
        raise NotImplementedError(
            "Brent has no symbolic jacobian; use convert_to_format='casadi'."
        )
