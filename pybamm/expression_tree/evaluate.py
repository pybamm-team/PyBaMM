#
# Simplify a symbol
#
import pybamm

import autograd.numpy as np
from collections import OrderedDict
import scipy.sparse
import copy


def id_to_python_variable(symbol_id, constant=False):
    if constant:
        var_format = "self.const_{:05d}"
    else:
        var_format = "self.var_{:05d}"
    return var_format.format(symbol_id).replace("-", "m")


def find_symbols(symbol, constant_symbols=OrderedDict(), variable_symbols=OrderedDict()):
    """
    This function converts an expression tree to a python function that acts like the
    tree's :func:`pybamm.Symbol.evaluate` function

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to a python function

    Returns
    -------
    str: name of python variable with result of execution

    """
    pybamm.logger.debug("Convert to python: {!s}".format(symbol))

    if symbol.is_constant():
        constant_symbols[symbol.id] = symbol.evaluate()
        return

    # process children
    for child in symbol.children:
        find_symbols(child, constant_symbols, variable_symbols)
    children_vars = [id_to_python_variable(child.id, child.is_constant())
                         for child in symbol.children]

    if isinstance(symbol, pybamm.BinaryOperator):
        left, right = symbol.children
        symbol_str = children_vars[0] + ' ' + symbol.name + ' ' + children_vars[1]

    elif isinstance(symbol, pybamm.UnaryOperator):
        if isinstance(symbol, pybamm.Function):
            constant_symbols[symbol.id] = symbol.func
            funct_var = id_to_python_variable(symbol.id, True)
            symbol_str = "{}({})".format(funct_var, children_vars[0])
        elif isinstance(symbol, pybamm.Index):
            symbol_str = "{}[{}]".format(children_vars[0],symbol.index)
        else:
            symbol_str = symbol.name + children_vars[0]

    elif isinstance(symbol, pybamm.Concatenation):

        if isinstance(symbol, pybamm.NumpyConcatenation):
            symbol_str = 'np.concatenate(({}))'.format(",".join(children_vars))

        elif isinstance(symbol, pybamm.SparseStack):
            symbol_str = "scipy.sparse.vstack(({}))".format(",".join(children_vars))

        elif isinstance(symbol, pybamm.DomainConcatenation):
            slice_starts = []
            child_vectors = []
            for child_var, slices in zip(children_vars, symbol._children_slices):
                for child_dom, child_slice in slices.items():
                    slice_starts.append(symbol._slices[child_dom].start)
                    child_vectors.append("{}[{}:{}]".format(
                        child_var, child_slice.start, child_slice.stop
                    ))
            child_vectors = [v for _, v in sorted(zip(slice_starts, child_vectors))]
            symbol_str = "np.concatenate(({}))".format(",".join(child_vectors))
        else:
            raise NotImplementedError

    elif isinstance(symbol, pybamm.StateVector):
        symbol_str = symbol.name

    elif isinstance(symbol, pybamm.Time):
        symbol_str = 't'

    else:
        raise NotImplementedError(
            "Not implemented for a symbol of type '{}'".format(type(symbol))
        )

    variable_symbols[symbol.id] = symbol_str


def to_python(symbol):
    """
    This function converts an expression tree to a python function that acts like the
    tree's :func:`pybamm.Symbol.evaluate` function

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to a python function

    Returns
    -------
    str: string representing a python function with signature func(t=None, y=None)

    """

    constant_values = OrderedDict()
    variable_symbols = OrderedDict()
    find_symbols(symbol, constant_values, variable_symbols)

    line_format = "{} = {}"
    variable_lines = [
        line_format.format(
            id_to_python_variable(symbol_id, False),
            symbol_line
        )
        for symbol_id, symbol_line in variable_symbols.items()
    ]

    return constant_values, "\n".join(variable_lines)


class EvaluatorPython:
    def __init__(self, symbol):
        constants, self._variable_function = pybamm.to_python(symbol)
        for symbol_id, value in constants.items():
            setattr(self, id_to_python_variable(symbol_id, True).replace("self.",""), value)
        self._result_var = id_to_python_variable(symbol.id, symbol.is_constant())
        self._variable_compiled = compile(
            self._variable_function, self._result_var, 'exec')
        self._return_compiled = compile(
            self._result_var, 'return'+self._result_var, 'eval')

    def evaluate(self, t=None, y=None):
        exec(self._variable_compiled)
        return eval(self._return_compiled)
