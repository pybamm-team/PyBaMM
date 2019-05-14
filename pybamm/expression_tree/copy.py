#
# Make a new copy of a symbol
#
import pybamm


def make_new_copy(symbol):
    """
    Make a new copy of a symbol, to avoid Tree corruption errors while bypassing
    copy.deepcopy(), which is slow.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to copy

    Returns
    -------
    :class:`pybamm.Symbol`
        New copy of the symbol
    """
    if isinstance(symbol, pybamm.BinaryOperator):
        # process children
        new_left = make_new_copy(symbol.left)
        new_right = make_new_copy(symbol.right)
        # make new symbol, ensure domain remains the same
        return symbol.__class__(new_left, new_right)

    elif isinstance(symbol, pybamm.UnaryOperator):
        new_child = make_new_copy(symbol.child)
        return symbol._unary_new_copy(new_child)

    elif isinstance(symbol, pybamm.Concatenation):
        new_children = [make_new_copy(child) for child in symbol.cached_children]
        return symbol._concatenation_new_copy(new_children)

    # Other cases: return new variable to avoid tree internal corruption
    elif isinstance(symbol, (pybamm.Parameter, pybamm.Variable)):
        return symbol.__class__(symbol.name, symbol.domain)

    elif isinstance(symbol, pybamm.StateVector):
        return pybamm.StateVector(symbol.y_slice, symbol.name)

    elif isinstance(symbol, pybamm.Scalar):
        return pybamm.Scalar(symbol.value, symbol.name, symbol.domain)

    elif isinstance(symbol, pybamm.Array):
        return symbol.__class__(
            symbol.entries, symbol.name, symbol.domain, symbol.entries_string
        )

    elif isinstance(symbol, pybamm.SpatialVariable):
        return pybamm.SpatialVariable(symbol.name, symbol.domain, symbol.coord_sys)

    elif isinstance(symbol, pybamm.Time):
        return pybamm.Time()

    else:
        raise NotImplementedError(
            "Cannot make new copy of symbol of type '{}'".format(type(symbol))
        )
