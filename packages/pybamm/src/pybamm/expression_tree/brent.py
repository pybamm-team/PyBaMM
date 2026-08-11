#
# A bracketed scalar rootfind as an expression tree node
#
from __future__ import annotations

import casadi
import numpy as np

import pybamm


class BrentUnknown(pybamm.Symbol):
    """
    A symbol to solve a :class:`Brent` node for.

    ``Brent`` accepts any symbol as its unknown, but a plain
    :class:`pybamm.Symbol` has no shape, a :class:`pybamm.Variable` must appear in
    ``rhs`` or ``algebraic`` to survive discretisation, and a
    :class:`pybamm.InputParameter` is collected as an input the caller then has to
    supply. This is bound by the enclosing ``Brent`` before any of that matters, so
    it passes through every pass untouched and is never asked for.
    """

    def _evaluate_for_shape(self):
        return np.nan * np.ones((1, 1))

    def create_copy(self, new_children=None, perform_simplifications=True):
        return BrentUnknown(self.name)


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
        The value being solved for. Any symbol will do, so long as it appears in
        ``residual`` and nowhere else in the surrounding expression;
        :class:`pybamm.BrentUnknown` is the one that survives every pass.
    bounds : tuple
        ``(lo, hi)``, the bracket to search. Either may be an expression. With
        ``max_expansions`` set these are only a starting scale, not a limit.
    guess : :class:`pybamm.Symbol` or float, optional
        Where to start. Only sets the scale of the first outward step, so it does
        not have to be close. Defaults to the midpoint of ``bounds``.
    max_expansions : int, optional
        Outward steps allowed when ``bounds`` holds no sign change. The default ``0``
        requires a valid bracket and raises without one. Positive walks outwards until
        the sign changes -- unambiguous for a monotonic residual -- and **never
        raises**: the closest point found comes back instead.
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
        guess: pybamm.Symbol | float | None = None,
        max_expansions: int = 0,
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
        if max_expansions < 0:
            raise ValueError(
                f"max_expansions must not be negative, got {max_expansions}"
            )
        self.abstol = abstol
        self.max_iter = max_iter
        self.max_expansions = max_expansions
        lo, hi = (pybamm.convert_to_symbol(bound) for bound in bounds)
        if guess is None:
            guess = (lo + hi) / 2
        # the unknown is a child so that it survives copying and serialisation with
        # the rest of the tree; it also appears inside the residual
        children = [residual, unknown, lo, hi, pybamm.convert_to_symbol(guess)]
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
        return tuple(self.children[2:4])

    @property
    def guess(self):
        """Where the search starts."""
        return self.children[4]

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        # The hyperparameters change the answer, so two nodes that differ only in them
        # are different symbols. Leaving them out lets the id-keyed caches in
        # `ParameterValues.process_symbol` and `_to_casadi` return the wrong one.
        super().set_id()
        self._id = hash((self._id, self.abstol, self.max_iter, self.max_expansions))

    def create_copy(self, new_children=None, perform_simplifications=True):
        residual, unknown, lo, hi, guess = self._children_for_copying(new_children)
        return Brent(
            residual,
            unknown,
            (lo, hi),
            guess=guess,
            max_expansions=self.max_expansions,
            abstol=self.abstol,
            max_iter=self.max_iter,
            name=self.name,
        )

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def to_json(self):
        json_dict = super().to_json()
        json_dict.update(
            {
                "abstol": self.abstol,
                "max_iter": self.max_iter,
                "max_expansions": self.max_expansions,
            }
        )
        return json_dict

    @classmethod
    def _from_json(cls, snippet: dict):
        residual, unknown, lo, hi, guess = snippet["children"]
        return cls(
            residual,
            unknown,
            (lo, hi),
            guess=guess,
            max_expansions=snippet["max_expansions"],
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
        lo, hi, guess = (
            child._to_casadi_inner(t, y, y_dot, inputs, casadi_symbols)
            for child in self.children[2:]
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
            {
                "abstol": self.abstol,
                "max_iter": self.max_iter,
                "max_expansions": self.max_expansions,
            },
        )
        return solver(guess, lo, hi, *free)

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
