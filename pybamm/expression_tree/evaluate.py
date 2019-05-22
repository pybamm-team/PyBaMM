#
# Simplify a symbol
#
import pybamm

import autograd.numpy as np
from collections import OrderedDict
import scipy.sparse
import copy


def id_to_python_variable(symbol_id):
    var_format = "var_{:05d}"
    return var_format.format(symbol_id).replace("-", "m")


def find_symbols(symbol, known_symbols=OrderedDict()):
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

    if isinstance(symbol, pybamm.BinaryOperator):
        left, right = symbol.children
        # process children
        find_symbols(left, known_symbols)
        find_symbols(right, known_symbols)
        left_var = id_to_python_variable(left.id)
        right_var = id_to_python_variable(right.id)
        symbol_str = left_var + ' ' + symbol.name + ' ' + right_var

    elif isinstance(symbol, pybamm.UnaryOperator):
        find_symbols(symbol.child, known_symbols)
        child_var = id_to_python_variable(symbol.child.id)
        symbol_str = symbol.name + child_var

    elif isinstance(symbol, pybamm.NumpyConcatenation):
        for child in symbol.children:
            find_symbols(child, known_symbols)
        children_vars = [id_to_python_variable(child.id) for child in symbol.children]
        symbol_str = 'np.concatenate([{}])'.format(",".join(children_var))

    elif isinstance(symbol, pybamm.StateVector):
        symbol_str = symbol.name

    elif isinstance(symbol, pybamm.Scalar):
        symbol_str = str(symbol.value)

    elif isinstance(symbol, pybamm.Array):
        if scipy.sparse.issparse(symbol.entries):
            if not isinstance(symbol.entries, scipy.sparse.csr_matrix):
                raise NotImplementedError
            data_str = "".join([
                "np.array([",
                ",".join([str(e) for e in symbol.entries.data]),
                "])"
            ])
            indices_str = "".join([
                "np.array([",
                ",".join([str(e) for e in symbol.entries.indices]),
                "])"
            ])
            indptr_str = "".join([
                "np.array([",
                ",".join([str(e) for e in symbol.entries.indptr]),
                "])"
            ])

            symbol_str = 'scipy.sparse.csr_matrix(({}, {}, {}))'\
                .format(data_str, indices_str, indptr_str)
        else:
            rows = [
                "[{}]".format(",".join([str(e) for e in row]))
                for row in symbol.entries
            ]
            matrix = "[{}]".format(",".join(rows))
            symbol_str = "np.array({})".format(matrix)

    elif isinstance(symbol, pybamm.Time):
        symbol_str = 't'

    else:
        raise NotImplementedError(
            "Not implemented for a symbol of type '{}'".format(type(symbol))
        )

    known_symbols[symbol.id] = symbol_str


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

    known_symbols = OrderedDict()
    find_symbols(symbol, known_symbols)

    line_format = "{} = {}"

    lines = [
        line_format.format(id_to_python_variable(symbol_id), symbol_line)
        for symbol_id, symbol_line in known_symbols.items()
    ]

    func_str = "\n".join(lines)

    return func_str


class EvaluatorPython:
    def __init__(self, symbol):
        funct_str = pybamm.to_python(symbol)
        self._result_var = id_to_python_variable(symbol.id)
        self._compiled_function = compile(funct_str, self._result_var, 'exec')

    def evaluate(self, t=None, y=None):
        exec(self._compiled_function)
        return eval(self._result_var)
