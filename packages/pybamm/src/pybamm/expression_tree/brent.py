#
# A bracketed scalar rootfind as an expression tree node
#
from __future__ import annotations

import casadi

import pybamm


class Brent(pybamm.Symbol):
    """
    Solve ``f == y_target`` for ``unknown`` in ``[lo, hi]``, by Brent's method.

    Every argument is an expression, so the target, the bracket, and anything inside
    ``f`` may be a :class:`pybamm.InputParameter` or any other symbol. Nothing is solved
    in Python: the node converts to a CasADi ``rootfinder`` using the native ``brent``
    plugin registered by ``pybammsolvers``, so the whole solve runs inside the CasADi
    graph.

    Brent needs only a sign change over the bracket, so it converges where a Newton
    iteration stalls, and the answer cannot leave ``[lo, hi]``. Derivatives come from
    CasADi's implicit function theorem, exactly.

    Parameters
    ----------
    f : :class:`pybamm.Symbol`
        The expression to invert. Must contain ``unknown``.
    unknown : :class:`pybamm.Symbol`
        The value being solved for. Any symbol will do, so long as it appears in ``f``
        and nowhere else in the surrounding expression.
    y_target : :class:`pybamm.Symbol` or float
        The value ``f`` must take.
    lo, hi : :class:`pybamm.Symbol` or float
        The bracket to search.
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
        sto = pybamm.Symbol("stoichiometry")
        node = pybamm.Brent(param.n.prim.U(sto, T), sto, voltage, 0, 1)
    """

    def __init__(
        self,
        f: pybamm.Symbol,
        unknown: pybamm.Symbol,
        y_target: pybamm.Symbol | float,
        lo: pybamm.Symbol | float,
        hi: pybamm.Symbol | float,
        abstol: float = 1e-14,
        max_iter: int = 100,
        name: str = "brent",
    ):
        if not isinstance(unknown, pybamm.Symbol):
            raise TypeError(
                f"unknown must be a pybamm.Symbol, got {type(unknown).__name__}"
            )
        if not isinstance(f, pybamm.Symbol):
            raise TypeError(f"f must be a pybamm.Symbol, got {type(f).__name__}")
        if not any(node == unknown for node in f.pre_order()):
            raise ValueError(f"'{unknown}' does not appear in '{f}'")
        self.unknown = unknown
        self.abstol = abstol
        self.max_iter = max_iter
        children = [
            f,
            *(
                value if isinstance(value, pybamm.Symbol) else pybamm.Scalar(value)
                for value in (y_target, lo, hi)
            ),
        ]
        super().__init__(name, children=children)

    def create_copy(self, new_children=None, perform_simplifications=True):
        children = self._children_for_copying(new_children)
        return Brent(
            children[0],
            self.unknown,
            children[1],
            children[2],
            children[3],
            abstol=self.abstol,
            max_iter=self.max_iter,
            name=self.name,
        )

    def _evaluate_for_shape(self):
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        unknown = casadi.MX.sym(f"brent_unknown_{abs(self.id)}")
        # the unknown resolves through the conversion cache; the copy keeps the binding
        # local to this node, so an enclosing conversion never sees it
        residual = self.children[0]._to_casadi_inner(
            t, y, y_dot, inputs, {**casadi_symbols, self.unknown: unknown}
        )
        target, lo, hi = (
            child._to_casadi_inner(t, y, y_dot, inputs, casadi_symbols)
            for child in self.children[1:]
        )

        equation = residual - target
        # Pass only the symbols the residual actually reads. Handing the oracle the
        # whole state vector would copy it into the solve on every evaluation.
        free = [s for s in casadi.symvar(equation) if not casadi.is_equal(s, unknown)]
        # the plugin reads the bracket from its own inputs rather than the residual, so
        # these are declared here and left unused
        lo_sym, hi_sym = casadi.MX.sym("lo"), casadi.MX.sym("hi")
        oracle = casadi.Function(
            f"brent_oracle_{abs(self.id)}", [unknown, lo_sym, hi_sym, *free], [equation]
        )
        solver = casadi.rootfinder(
            f"brent_{abs(self.id)}",
            "brent",
            oracle,
            {
                "lo_index": 1,
                "hi_index": 2,
                "abstol": self.abstol,
                "max_iter": self.max_iter,
            },
        )
        # the guess is required by the rootfinder interface and ignored by a bracketed
        # method, so either end of the bracket does
        return solver(lo, lo, hi, *free)

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
