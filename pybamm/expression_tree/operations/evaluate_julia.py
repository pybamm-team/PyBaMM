#
# Write a symbol to Julia
#
import pybamm

import numpy as np
import scipy.sparse
from collections import OrderedDict

import numbers


def id_to_julia_variable(symbol_id, constant=False):
    """
    This function defines the format for the julia variable names used in find_symbols
    and to_julia. Variable names are based on a nodes' id to make them unique
    """

    if constant:
        var_format = "const_{:05d}"
    else:
        var_format = "var_{:05d}"

    # Need to replace "-" character to make them valid julia variable names
    return var_format.format(symbol_id).replace("-", "m")


def find_symbols(symbol, constant_symbols, variable_symbols, to_dense=False):
    """
    This function converts an expression tree to a dictionary of node id's and strings
    specifying valid julia code to calculate that nodes value, given y and t.

    The function distinguishes between nodes that represent constant nodes in the tree
    (e.g. a pybamm.Matrix), and those that are variable (e.g. subtrees that contain
    pybamm.StateVector). The former are put in `constant_symbols`, the latter in
    `variable_symbols`

    Note that it is important that the arguments `constant_symbols` and
    `variable_symbols` be and *ordered* dict, since the final ordering of the code lines
    are important for the calculations. A dict is specified rather than a list so that
    identical subtrees (which give identical id's) are not recalculated in the code

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol or expression tree to convert

    constant_symbol: collections.OrderedDict
        The output dictionary of constant symbol ids to lines of code

    variable_symbol: collections.OrderedDict
        The output dictionary of variable (with y or t) symbol ids to lines of code

    to_dense: bool
        If True, all constants and expressions are converted to using dense matrices

    """
    if symbol.is_constant():
        value = symbol.evaluate()
        if not isinstance(value, numbers.Number):
            if scipy.sparse.issparse(value):
                # Create Julia SparseArray
                row, col, data = scipy.sparse.find(value)
                m, n = value.shape
                # add 1 to correct for 1-indexing in Julia
                # use array2string so that commas are included
                constant_symbols[symbol.id] = "sparse({}, {}, {}, {}, {})".format(
                    np.array2string(row + 1, separator=","),
                    np.array2string(col + 1, separator=","),
                    np.array2string(data, separator=","),
                    m,
                    n,
                )
            else:
                constant_symbols[symbol.id] = value
        return

    # process children recursively
    for child in symbol.children:
        find_symbols(child, constant_symbols, variable_symbols, to_dense)

    # calculate the variable names that will hold the result of calculating the
    # children variables
    children_vars = []
    for child in symbol.children:
        if child.is_constant():
            child_eval = child.evaluate()
            if isinstance(child_eval, numbers.Number):
                children_vars.append(str(child_eval))
            else:
                children_vars.append(id_to_julia_variable(child.id, True))
        else:
            children_vars.append(id_to_julia_variable(child.id, False))

    if isinstance(symbol, pybamm.BinaryOperator):
        # Multiplication and Division need special handling for scipy sparse matrices
        # TODO: we can pass through a dummy y and t to get the type and then hardcode
        # the right line, avoiding these checks
        if isinstance(symbol, (pybamm.Multiplication, pybamm.Inner)):
            symbol_str = "{0} .* {1}".format(children_vars[0], children_vars[1])
        if isinstance(symbol, pybamm.MatrixMultiplication):
            symbol_str = "{0} * {1}".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Division):
            symbol_str = "{0} ./ {1}".format(children_vars[0], children_vars[1])

        elif isinstance(symbol, pybamm.Minimum):
            symbol_str = "np.minimum({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Maximum):
            symbol_str = "np.maximum({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Power):
            symbol_str = children_vars[0] + " ^ " + children_vars[1]
        else:
            symbol_str = children_vars[0] + " " + symbol.name + " " + children_vars[1]

    elif isinstance(symbol, pybamm.UnaryOperator):
        # Index has a different syntax than other univariate operations
        if isinstance(symbol, pybamm.Index):
            # Because of how julia indexing works, add 1 to the start, but not to the
            # stop
            symbol_str = "{}[{}:{}]".format(
                children_vars[0], symbol.slice.start + 1, symbol.slice.stop
            )
        else:
            symbol_str = symbol.name + children_vars[0]

    elif isinstance(symbol, pybamm.Function):
        children_str = ""
        for child_var in children_vars:
            if children_str == "":
                children_str = child_var
            else:
                children_str += ", " + child_var
        # write functions directly
        julia_name = symbol.julia_name
        # add a . to allow elementwise operations
        symbol_str = "{}.({})".format(julia_name, children_str)

    elif isinstance(symbol, pybamm.Concatenation):

        # don't bother to concatenate if there is only a single child
        if isinstance(symbol, (pybamm.NumpyConcatenation, pybamm.SparseStack)):
            if len(children_vars) == 1:
                symbol_str = children_vars
            else:
                symbol_str = "vcat({})".format(",".join(children_vars))

        # DomainConcatenation specifies a particular ordering for the concatenation,
        # which we must follow
        elif isinstance(symbol, pybamm.DomainConcatenation):
            slice_starts = []
            all_child_vectors = []
            for i in range(symbol.secondary_dimensions_npts):
                child_vectors = []
                for child_var, slices in zip(children_vars, symbol._children_slices):
                    for child_dom, child_slice in slices.items():
                        slice_starts.append(symbol._slices[child_dom][i].start)
                        child_vectors.append(
                            "{}[{}:{}]".format(
                                child_var, child_slice[i].start, child_slice[i].stop
                            )
                        )
                all_child_vectors.extend(
                    [v for _, v in sorted(zip(slice_starts, child_vectors))]
                )
            if len(children_vars) > 1 or symbol.secondary_dimensions_npts > 1:
                symbol_str = "np.concatenate(({}))".format(",".join(all_child_vectors))
            else:
                symbol_str = "{}".format(",".join(children_vars))
        else:
            raise NotImplementedError

    # Note: we assume that y is being passed as a column vector
    elif isinstance(symbol, pybamm.StateVector):
        indices = np.argwhere(symbol.evaluation_array).reshape(-1).astype(np.int32)
        # add 1 since julia uses 1-indexing
        indices += 1
        consecutive = np.all(indices[1:] - indices[:-1] == 1)
        if len(indices) == 1:
            symbol_str = "y[{}]".format(indices[0])
        elif consecutive:
            # julia does include the final value
            symbol_str = "y[{}:{}]".format(indices[0], indices[-1])
        else:
            indices_array = pybamm.Array(indices)
            constant_symbols[indices_array.id] = indices
            index_name = id_to_julia_variable(indices_array.id, True)
            symbol_str = "y[{}]".format(index_name)

    elif isinstance(symbol, pybamm.Time):
        symbol_str = "t"

    elif isinstance(symbol, pybamm.InputParameter):
        symbol_str = "inputs['{}']".format(symbol.name)

    else:
        raise NotImplementedError(
            "Conversion to Julia not implemented for a symbol of type '{}'".format(
                type(symbol)
            )
        )

    variable_symbols[symbol.id] = symbol_str


def to_julia(symbol, debug=False, to_dense=False):
    """
    This function converts an expression tree into a dict of constant input values, and
    valid julia code that acts like the tree's :func:`pybamm.Symbol.evaluate` function

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to julia code

    debug : bool
        If set to True, the function also emits debug code

    Returns
    -------
    collections.OrderedDict:
        dict mapping node id to a constant value. Represents all the constant nodes in
        the expression tree
    str:
        valid julia code that will evaluate all the variable nodes in the tree.
    to_dense: bool
        If True, all constants and expressions are converted to using dense matrices

    """

    constant_values = OrderedDict()
    variable_symbols = OrderedDict()
    find_symbols(symbol, constant_values, variable_symbols, to_dense)

    line_format = "{} = {}"

    if debug:
        variable_lines = [
            "print('{}'); ".format(
                line_format.format(id_to_julia_variable(symbol_id, False), symbol_line)
            )
            + line_format.format(id_to_julia_variable(symbol_id, False), symbol_line)
            + "; print(type({0}),{0}.shape)".format(
                id_to_julia_variable(symbol_id, False)
            )
            for symbol_id, symbol_line in variable_symbols.items()
        ]
    else:
        variable_lines = [
            line_format.format(id_to_julia_variable(symbol_id, False), symbol_line)
            for symbol_id, symbol_line in variable_symbols.items()
        ]

    return constant_values, "\n".join(variable_lines)


def get_julia_function(symbol):
    """
    Converts a pybamm expression tree into pure julia code that will calculate the
    result of calling `evaluate(t, y)` on the given expression tree.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to julia code

    Returns
    -------
    julia_str : str
        String of julia code, to be evaluated by ``julia.Main.eval``

    """

    constants, julia_str = to_julia(symbol, debug=False)

    # extract constants in generated function
    for symbol_id, const_value in constants.items():
        const_name = id_to_julia_variable(symbol_id, True)
        julia_str = "{} = {}\n".format(const_name, const_value) + julia_str

    # indent code
    julia_str = "   " + julia_str
    julia_str = julia_str.replace("\n", "\n   ")

    # add function def and sparse arrays to first line
    imports = "using SparseArrays\n"
    julia_str = imports + "function f_pybamm(t, y, p)\n" + julia_str

    # calculate the final variable that will output the result
    result_var = id_to_julia_variable(symbol.id, symbol.is_constant())
    if symbol.is_constant():
        result_value = symbol.evaluate()

    # add return line
    if symbol.is_constant() and isinstance(result_value, numbers.Number):
        julia_str = julia_str + "\n   return " + str(result_value) + "\nend"
    else:
        julia_str = julia_str + "\n   return " + result_var + "\nend"

    return julia_str

