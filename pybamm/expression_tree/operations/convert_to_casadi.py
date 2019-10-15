#
# Convert a PyBaMM expression tree to a CasADi expression tree
#
import pybamm
import casadi


class CasadiConverter(object):
    def __init__(self, casadi_symbols=None):
        self._casadi_symbols = casadi_symbols or {}

    def convert(self, symbol):
        """
        This function recurses down the tree, applying any simplifications defined in
        classes derived from pybamm.Symbol. E.g. any expression multiplied by a
        pybamm.Scalar(0) will be simplified to a pybamm.Scalar(0).
        If a symbol has already been simplified, the stored value is returned.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to convert

        Returns
        -------
        CasADi symbol
            The convert symbol
        """

        try:
            return self._casadi_symbols[symbol.id]
        except KeyError:
            casadi_symbol = self._convert(symbol)
            self._casadi_symbols[symbol.id] = casadi_symbol

            return casadi_symbol

    def _convert(self, symbol):
        """ See :meth:`Simplification.convert()`. """
        if isinstance(symbol, pybamm.Scalar):
            return casadi.SX(symbol.evaluate())

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            converted_left = self.convert(left)
            converted_right = self.convert(right)
            # _binary_evaluate defined in derived classes for specific rules
            return symbol._binary_evaluate(converted_left, converted_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            converted_child = self.convert(symbol.child)
            if isinstance(symbol, pybamm.AbsoluteValue):
                return casadi.fabs(converted_child)
            return symbol._unary_evaluate(converted_child)

        elif isinstance(symbol, pybamm.Function):
            converted_children = [None] * len(symbol.children)
            for i, child in enumerate(symbol.children):
                converted_children[i] = self.convert(child)
            return symbol._function_evaluate(converted_children)

        elif isinstance(symbol, pybamm.Concatenation):
            converted_children = [self.convert(child) for child in symbol.children]
            return symbol._concatenation_evaluate(converted_children)

        else:
            raise TypeError(
                """
                Cannot convert symbol of type '{}' to CasADi. Symbols must all be
                'linear algebra' at this stage.
                """.format(
                    type(symbol)
                )
            )
