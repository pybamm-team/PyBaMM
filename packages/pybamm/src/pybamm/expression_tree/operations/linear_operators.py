#
# Conversion to CasADi with linear operators folded and input-only parts hoisted
#
from __future__ import annotations

from enum import Enum

import casadi

import pybamm


class _Kind(Enum):
    # Depends on neither the states nor time, so only on inputs and constants
    PARAMETRIC = 1
    # Homogeneous linear in the states, with parametric coefficients
    LINEAR = 2
    OTHER = 3


class CasadiConversion:
    """
    CasADi conversion of discretised equations that a solver evaluates many times
    with the same inputs.

    A subexpression homogeneous linear in the states, with input-only
    coefficients, becomes one sparse product ``A @ y``. Input-only
    subexpressions, such matrices included, become symbols whose values are in
    :attr:`precomputed`, for the solver to compute once per solve.

    Parameters
    ----------
    jacobian_sparsity : :class:`pybamm.JacobianSparsity`
        Structural sparsity of the expressions' Jacobians.
    t, y : :class:`casadi.MX`
        Time and the state vector.
    inputs : dict
        CasADi symbols of the inputs, by name.
    """

    def __init__(
        self,
        jacobian_sparsity: pybamm.JacobianSparsity,
        t: casadi.MX,
        y: casadi.MX,
        inputs: dict,
    ):
        #: Structural sparsity of the expressions' Jacobians
        self.jacobian_sparsity = jacobian_sparsity
        self._t, self._y, self._inputs = t, y, inputs
        self._kinds: dict[int, _Kind] = {}
        self._has_product: dict[int, bool] = {}
        self._has_input: dict[int, bool] = {}
        self._casadi: dict[pybamm.Symbol, casadi.MX] = {}
        # Conversions in the inputs alone, with no stand-ins
        self._casadi_in_inputs: dict[pybamm.Symbol, casadi.MX] = {}
        self._visited: set[int] = set()
        #: Column symbols of the nonzeros of input-only values, and those nonzeros
        #: as expressions in the inputs
        self.precomputed: list[tuple[casadi.MX, casadi.MX]] = []
        self._recorded: dict[str, casadi.Function] = {}

    def to_casadi(self, symbol: pybamm.Symbol) -> casadi.MX:
        """
        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The discretised expression.

        Returns
        -------
        :class:`casadi.MX`
            The expression in time, the states and the symbols of
            :attr:`precomputed`.
        """
        self._prepare(symbol)
        return symbol.to_casadi(
            self._t, self._y, inputs=self._inputs, casadi_symbols=self._casadi
        )

    def function(
        self, name: str, arguments: list[casadi.MX], outputs: list[casadi.MX]
    ) -> casadi.Function:
        """
        A function of expressions from :meth:`to_casadi`.

        Parameters
        ----------
        name : str
            The function's name.
        arguments : list of :class:`casadi.MX`
            Time, the states, the stacked inputs, then any others.
        outputs : list of :class:`casadi.MX`
            The expressions.

        Returns
        -------
        :class:`casadi.Function`
            The function of ``arguments``, which computes the input-only values
            it needs. The variant that reads them after the stacked inputs is kept
            for :meth:`precomputed_functions`.
        """
        extended = casadi.vertcat(
            arguments[2], *(symbol for symbol, _ in self.precomputed)
        )
        recorded = casadi.Function(
            name, [*arguments[:2], extended, *arguments[3:]], outputs
        )
        self._recorded[name] = recorded
        if not self.precomputed:
            return recorded
        values = casadi.vertcat(arguments[2], *(value for _, value in self.precomputed))
        return casadi.Function(
            name, arguments, recorded.call([*arguments[:2], values, *arguments[3:]])
        )

    def precomputed_functions(self, names: list[str]) -> dict[str, casadi.Function]:
        """
        The named functions that read the input-only values after the stacked
        inputs, and ``precompute``, from the stacked inputs to those values.

        Parameters
        ----------
        names : list of str
            Names given to :meth:`function`.

        Returns
        -------
        dict
            Functions by name, and ``precompute``. Empty if nothing is
            precomputed.
        """
        if not self.precomputed:
            return {}
        functions = {name: self._recorded[name] for name in names}
        functions["precompute"] = casadi.Function(
            "precompute",
            [casadi.vertcat(*self._inputs.values())],
            [casadi.vertcat(*(value for _, value in self.precomputed))],
        )
        return functions

    def _prepare(self, symbol: pybamm.Symbol, fold: bool = True):
        """Fold the linear operators (if ``fold``) and hoist the input-only parts
        of ``symbol``, outermost first."""
        stack = [symbol]
        while stack:
            node = stack.pop()
            if node.id in self._visited:
                continue
            self._visited.add(node.id)
            if isinstance(node, pybamm.Conditional):
                # Its branches become functions of time, the states and the inputs
                # alone, so nothing inside may stand for a precomputed value
                self._casadi[node] = node.to_casadi(
                    self._t,
                    self._y,
                    inputs=self._inputs,
                    casadi_symbols=self._casadi_in_inputs,
                )
                continue
            kind = self._kind(node)
            if kind is _Kind.PARAMETRIC:
                if self._has_input[node.id] and node.children:
                    self._hoist(node)
                continue
            if fold and kind is _Kind.LINEAR and self._has_product[node.id]:
                self._fold(node)
                continue
            stack.extend(node.children)

    def _hoist(self, node: pybamm.Symbol):
        value = node.to_casadi(
            self._t, self._y, inputs=self._inputs, casadi_symbols=self._casadi_in_inputs
        )
        self._casadi[node] = self._register(value)

    def _fold(self, operator: pybamm.Symbol):
        try:
            pattern = self.jacobian_sparsity.pattern(operator)
        except NotImplementedError:
            for child in operator.children:
                self._prepare(child)
            return
        # Hoist the operator's input-only coefficients before converting it
        for child in operator.children:
            self._prepare(child, fold=False)
        expression = operator.to_casadi(
            self._t, self._y, inputs=self._inputs, casadi_symbols=self._casadi
        )
        matrix = pybamm.casadi_jacobian(expression, self._y, pattern)
        if casadi.symvar(matrix):
            matrix = self._register(matrix)
        else:
            matrix = casadi.MX(casadi.Function("matrix", [], [matrix])()["o0"])
        self._casadi[operator] = casadi.mtimes(matrix, self._y)

    def _register(self, value: casadi.MX) -> casadi.MX:
        """Stand-in for an input-only value, which may refer to earlier ones."""
        nonzeros = casadi.reshape(value.nz[:], -1, 1)
        if self.precomputed:
            # In the inputs only, so precompute evaluates it directly
            symbols, values = zip(*self.precomputed, strict=True)
            nonzeros = casadi.substitute([nonzeros], list(symbols), list(values))[0]
        symbol = casadi.MX.sym(f"precomputed_{len(self.precomputed)}", value.nnz())
        self.precomputed.append((symbol, nonzeros))
        return casadi.MX(value.sparsity(), symbol)

    def _kind(self, symbol: pybamm.Symbol) -> _Kind:
        if symbol.id not in self._kinds:
            kind = self._classify(symbol)
            children = symbol.children
            self._kinds[symbol.id] = kind
            self._has_input[symbol.id] = isinstance(
                symbol, pybamm.InputParameter
            ) or any(self._has_input[child.id] for child in children)
            self._has_product[symbol.id] = kind is _Kind.LINEAR and (
                isinstance(symbol, pybamm.MatrixMultiplication)
                or any(self._has_product[child.id] for child in children)
            )
        return self._kinds[symbol.id]

    def _classify(self, symbol: pybamm.Symbol) -> _Kind:
        if isinstance(symbol, pybamm.StateVector):
            return _Kind.LINEAR
        if isinstance(symbol, (pybamm.Time, pybamm.StateVectorDot)):
            return _Kind.OTHER
        if not symbol.children:
            return _Kind.PARAMETRIC
        kinds = [self._kind(child) for child in symbol.children]
        if all(kind is _Kind.PARAMETRIC for kind in kinds):
            return _Kind.PARAMETRIC
        linear, parametric = _Kind.LINEAR, _Kind.PARAMETRIC
        if isinstance(symbol, pybamm.MatrixMultiplication):
            is_linear = kinds == [parametric, linear]
        elif isinstance(symbol, pybamm.Multiplication):
            is_linear = sorted(kind.value for kind in kinds) == [
                parametric.value,
                linear.value,
            ]
        elif isinstance(symbol, pybamm.Division):
            is_linear = kinds == [linear, parametric]
        elif isinstance(
            symbol,
            (
                pybamm.Addition,
                pybamm.Subtraction,
                pybamm.Negate,
                pybamm.Index,
                pybamm.NumpyConcatenation,
                pybamm.DomainConcatenation,
                pybamm.SparseStack,
            ),
        ):
            is_linear = all(kind is linear for kind in kinds)
        else:
            is_linear = False
        return linear if is_linear else _Kind.OTHER
