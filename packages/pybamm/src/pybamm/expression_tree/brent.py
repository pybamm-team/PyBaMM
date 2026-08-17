#
# A bracketed scalar rootfind as an expression tree node
#
from __future__ import annotations

import itertools

import casadi
import numpy as np

import pybamm


class BrentUnknown(pybamm.Symbol):
    """
    The scalar a :class:`Brent` solves for.

    Bound by the rootfinder, not by the model, so it is deliberately not a
    :class:`pybamm.Variable`: the checks that enumerate model states must not count it
    as one, or a ``Brent`` inside ``model.rhs`` or ``model.algebraic`` looks like an
    extra unknown with no equation. Each instance is uniquely named, so two ``Brent``
    nodes never share an unknown.

    Parameters
    ----------
    name : str, optional
        Name of the node. A unique suffix is appended.
    """

    _count = itertools.count()

    def __init__(self, name: str = "brent unknown"):
        super().__init__(f"{name} {next(BrentUnknown._count)}")

    def create_copy(self, new_children=None, perform_simplifications=True):
        # a copy must stay the *same* unknown, or the binding in Brent._to_casadi and
        # the residual would refer to two different symbols
        copy = BrentUnknown.__new__(BrentUnknown)
        pybamm.Symbol.__init__(copy, self.name)
        return copy

    def _evaluate_for_shape(self):
        # a scalar, but shaped like every other column-vector node so that the
        # broadcasting helpers can read shape[1]
        return np.nan * np.ones((1, 1))

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        # set by Brent._base_evaluate while it iterates; there is nothing to read
        # otherwise, since the value only exists inside a solve
        return self._value

    # a 1x1 column, the shape every other node in an equation reports
    _value = np.nan * np.ones((1, 1))


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
    unknown : :class:`pybamm.Symbol`
        The value being solved for. Must appear in ``residual`` and nowhere else in the
        surrounding expression. Use :class:`pybamm.BrentUnknown`, which is what the
        model checks recognise as bound rather than as a state; a
        :class:`pybamm.Variable` works for a standalone expression but makes the model
        look underdetermined once the node is inside ``rhs`` or ``algebraic``.
    bounds : tuple
        ``(lo, hi)``, the bracket to search. Either may be an expression.
    abstol : float, optional
        Absolute tolerance on the unknown. A hyperparameter: fixed when the node is
        built, not solved over.
    max_iter : int, optional
        Iteration cap. A hyperparameter, as ``abstol``.
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
        unknown: pybamm.Symbol,
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
        if not isinstance(unknown, pybamm.Symbol):
            raise TypeError(
                f"unknown must be a pybamm.Symbol, got {type(unknown).__name__}"
            )
        if not any(node == unknown for node in residual.pre_order()):
            raise ValueError(f"'{unknown}' does not appear in '{residual}'")
        if len(bounds) != 2:
            raise ValueError(f"bounds must be a (lo, hi) pair, got {bounds}")
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

    @property
    def residual(self):
        """The expression being driven to zero."""
        return self.children[0]

    @property
    def unknown(self):
        """The symbol being solved for."""
        return self.children[1]

    @property
    def bounds(self):
        """The bracket, as a ``(lo, hi)`` pair of symbols."""
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
        # the unknown resolves through the conversion cache; the copy keeps the binding
        # local to this node, so an enclosing conversion never sees it
        equation = self.children[0]._to_casadi_inner(
            t, y, y_dot, inputs, {**casadi_symbols, self.unknown: unknown}
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
        """Solve in NumPy, for the paths that do not go through CasADi.

        The CasADi conversion is the one that matters for solving; this exists so that
        shape inference and the NumPy evaluation path work on an expression containing
        a Brent node.
        """
        from scipy.optimize import brentq

        scalar = lambda v: np.asarray(v, dtype=float).reshape(-1)[0]
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

        try:
            root = brentq(g, lo, hi, xtol=self.abstol, maxiter=self.max_iter)
            return np.array([[root]])
        except (ValueError, RuntimeError):
            # a shape probe passes NaN through the bounds, and a bracket that does not
            # straddle a root has no answer to give
            return np.nan * np.ones((1, 1))
        finally:
            unknown._value = np.nan * np.ones((1, 1))

    def diff(self, variable):
        raise NotImplementedError(
            "Brent has no symbolic derivative. Its derivative comes from CasADi's "
            "implicit function theorem, so use convert_to_format='casadi'."
        )

    def _jac(self, variable):
        raise NotImplementedError(
            "Brent has no symbolic jacobian. Its derivative comes from CasADi's "
            "implicit function theorem, so use convert_to_format='casadi'."
        )
