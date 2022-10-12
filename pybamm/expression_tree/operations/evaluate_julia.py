#
# Write a symbol to Julia
#
import pybamm

import numpy as np
import scipy.sparse
from collections import OrderedDict
from pybamm.util import is_constant_and_can_evaluate

import numbers


def id_to_julia_variable(symbol_id, prefix):
    """
    This function defines the format for the julia variable names used in find_symbols
    and to_julia. Variable names are based on a nodes' id to make them unique
    """
    var_format = prefix + "_{:05d}"
    # Need to replace "-" character to make them valid julia variable names
    return var_format.format(symbol_id).replace("-", "m")


def find_symbols(
    symbol,
    constant_symbols,
    variable_symbols,
    variable_symbol_sizes,
    round_constants=True,
):
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

    constant_symbol : collections.OrderedDict
        The output dictionary of constant symbol ids to lines of code

    variable_symbol : collections.OrderedDict
        The output dictionary of variable (with y or t) symbol ids to lines of code

    variable_symbol_sizes : collections.OrderedDict
        The output dictionary of variable (with y or t) symbol ids to size of that
        variable, for caching

    """
    # ignore broadcasts for now
    if isinstance(symbol, pybamm.Broadcast):
        symbol = symbol.child
    if is_constant_and_can_evaluate(symbol):
        value = symbol.evaluate()
        if round_constants:
            value = np.round(value, decimals=11)
        if not isinstance(value, numbers.Number):
            if scipy.sparse.issparse(value):
                # Create Julia SparseArray
                row, col, data = scipy.sparse.find(value)
                if round_constants:
                    data = np.round(data, decimals=11)
                m, n = value.shape
                # Set print options large enough to avoid ellipsis
                # at least as big is len(row) = len(col) = len(data)
                np.set_printoptions(
                    threshold=max(np.get_printoptions()["threshold"], len(row) + 10)
                )
                # increase precision for printing
                np.set_printoptions(precision=20)
                # add 1 to correct for 1-indexing in Julia
                # use array2string so that commas are included
                constant_symbols[symbol.id] = "sparse({}, {}, {}, {}, {})".format(
                    np.array2string(row + 1, separator=","),
                    np.array2string(col + 1, separator=","),
                    np.array2string(data, separator=","),
                    m,
                    n,
                )
            elif value.shape == (1, 1):
                # Extract value if array has only one entry
                constant_symbols[symbol.id] = value[0, 0]
                variable_symbol_sizes[symbol.id] = 1
            elif value.shape[1] == 1:
                # Set print options large enough to avoid ellipsis
                # at least as big as len(row) = len(col) = len(data)
                np.set_printoptions(
                    threshold=max(
                        np.get_printoptions()["threshold"], value.shape[0] + 10
                    )
                )
                # Flatten a 1D array
                constant_symbols[symbol.id] = np.array2string(
                    value.flatten(), separator=","
                )
                variable_symbol_sizes[symbol.id] = symbol.shape[0]
            else:
                constant_symbols[symbol.id] = value
                # No need to save the size as it will not need to be used
        return

    # process children recursively
    for child in symbol.children:
        find_symbols(
            child,
            constant_symbols,
            variable_symbols,
            variable_symbol_sizes,
            round_constants=round_constants,
        )

    # calculate the variable names that will hold the result of calculating the
    # children variables
    children_vars = []
    for child in symbol.children:
        if isinstance(child, pybamm.Broadcast):
            child = child.child
        if is_constant_and_can_evaluate(child):
            child_eval = child.evaluate()
            if isinstance(child_eval, numbers.Number):
                children_vars.append(str(child_eval))
            else:
                children_vars.append(id_to_julia_variable(child.id, "const"))
        else:
            children_vars.append(id_to_julia_variable(child.id, "cache"))

    if isinstance(symbol, pybamm.BinaryOperator):
        # TODO: we can pass through a dummy y and t to get the type and then hardcode
        # the right line, avoiding these checks
        if isinstance(symbol, pybamm.MatrixMultiplication):
            symbol_str = "{0} @ {1}".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Inner):
            symbol_str = "{0} * {1}".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Minimum):
            symbol_str = "min({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Maximum):
            symbol_str = "max({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Power):
            # julia uses ^ instead of ** for power
            # include dot for elementwise operations
            symbol_str = children_vars[0] + " .^ " + children_vars[1]
        else:
            # all other operations use the same symbol
            symbol_str = children_vars[0] + " " + symbol.name + " " + children_vars[1]

    elif isinstance(symbol, pybamm.UnaryOperator):
        # Index has a different syntax than other univariate operations
        if isinstance(symbol, pybamm.Index):
            # Because of how julia indexing works, add 1 to the start, but not to the
            # stop
            symbol_str = "{}[{}:{}]".format(
                children_vars[0], symbol.slice.start + 1, symbol.slice.stop
            )
        elif isinstance(symbol, pybamm.Gradient):
            symbol_str = "grad_{}({})".format(tuple(symbol.domain), children_vars[0])
        elif isinstance(symbol, pybamm.Divergence):
            symbol_str = "div_{}({})".format(tuple(symbol.domain), children_vars[0])
        elif isinstance(symbol, pybamm.Broadcast):
            # ignore broadcasts for now
            symbol_str = children_vars[0]
        elif isinstance(symbol, pybamm.BoundaryValue):
            symbol_str = "boundary_value_{}({})".format(symbol.side, children_vars[0])
        else:
            symbol_str = symbol.name + children_vars[0]

    elif isinstance(symbol, pybamm.Function):
        # write functions directly
        symbol_str = "{}({})".format(symbol.julia_name, ", ".join(children_vars))

    elif isinstance(symbol, (pybamm.Variable, pybamm.ConcatenationVariable)):
        # No need to do anything if a Variable is found
        return

    elif isinstance(symbol, pybamm.Concatenation):
        if isinstance(symbol, (pybamm.NumpyConcatenation, pybamm.SparseStack)):
            # return a list of the children variables, which will be converted to a
            # line by line assignment
            # return this as a string so that other functionality still works
            # also save sizes
            symbol_str = "["
            for child in children_vars:
                child_id = child[6:].replace("m", "-")
                size = variable_symbol_sizes[int(child_id)]
                symbol_str += "{}::{}, ".format(size, child)
            symbol_str = symbol_str[:-2] + "]"

        # DomainConcatenation specifies a particular ordering for the concatenation,
        # which we must follow
        elif isinstance(symbol, pybamm.DomainConcatenation):
            if symbol.secondary_dimensions_npts == 1:
                all_child_vectors = children_vars
                all_child_sizes = [
                    variable_symbol_sizes[int(child[6:].replace("m", "-"))]
                    for child in children_vars
                ]
            else:
                slice_starts = []
                all_child_vectors = []
                all_child_sizes = []
                for i in range(symbol.secondary_dimensions_npts):
                    child_vectors = []
                    child_sizes = []
                    for child_var, slices in zip(
                        children_vars, symbol._children_slices
                    ):
                        for child_dom, child_slice in slices.items():
                            slice_starts.append(symbol._slices[child_dom][i].start)
                            # add 1 to slice start to account for julia indexing
                            child_vectors.append(
                                "@view {}[{}:{}]".format(
                                    child_var,
                                    child_slice[i].start + 1,
                                    child_slice[i].stop,
                                )
                            )
                            child_sizes.append(
                                child_slice[i].stop - child_slice[i].start
                            )
                    all_child_vectors.extend(
                        [v for _, v in sorted(zip(slice_starts, child_vectors))]
                    )
                    all_child_sizes.extend(
                        [v for _, v in sorted(zip(slice_starts, child_sizes))]
                    )
            # return a list of the children variables, which will be converted to a
            # line by line assignment
            # return this as a string so that other functionality still works
            # also save sizes
            symbol_str = "["
            for child, size in zip(all_child_vectors, all_child_sizes):
                symbol_str += "{}::{}, ".format(size, child)
            symbol_str = symbol_str[:-2] + "]"

        else:
            # A regular Concatenation for the MTK model
            # We will define the concatenation function separately
            symbol_str = "concatenation(" + ", ".join(children_vars) + ")"

    # Note: we assume that y is being passed as a column vector
    elif isinstance(symbol, pybamm.StateVectorBase):
        if isinstance(symbol, pybamm.StateVector):
            name = "@view y"
        elif isinstance(symbol, pybamm.StateVectorDot):
            name = "@view dy"
        indices = np.argwhere(symbol.evaluation_array).reshape(-1).astype(np.int32)
        # add 1 since julia uses 1-indexing
        indices += 1
        if len(indices) == 1:
            symbol_str = "{}[{}]".format(name, indices[0])
        else:
            # julia does include the final value
            symbol_str = "{}[{}:{}]".format(name, indices[0], indices[-1])

    elif isinstance(symbol, pybamm.Time):
        symbol_str = "t"

    elif isinstance(symbol, pybamm.InputParameter):
        symbol_str = "inputs['{}']".format(symbol.name)

    elif isinstance(symbol, pybamm.SpatialVariable):
        symbol_str = symbol.name

    elif isinstance(symbol, pybamm.FunctionParameter):
        symbol_str = "{}({})".format(symbol.name, ", ".join(children_vars))

    else:
        raise NotImplementedError(
            "Conversion to Julia not implemented for a symbol of type '{}'".format(
                type(symbol)
            )
        )

    variable_symbols[symbol.id] = symbol_str

    # Save the size of the symbol
    try:
        if symbol.shape == ():
            variable_symbol_sizes[symbol.id] = 1
        else:
            variable_symbol_sizes[symbol.id] = symbol.shape[0]
    except NotImplementedError:
        pass


def to_julia(symbol, round_constants=True):
    """
    This function converts an expression tree into a dict of constant input values, and
    valid julia code that acts like the tree's :func:`pybamm.Symbol.evaluate` function

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to julia code

    Returns
    -------
    constant_values : collections.OrderedDict
        dict mapping node id to a constant value. Represents all the constant nodes in
        the expression tree
    str
        valid julia code that will evaluate all the variable nodes in the tree.

    """

    constant_values = OrderedDict()
    variable_symbols = OrderedDict()
    variable_symbol_sizes = OrderedDict()
    find_symbols(
        symbol,
        constant_values,
        variable_symbols,
        variable_symbol_sizes,
        round_constants=round_constants,
    )

    return constant_values, variable_symbols, variable_symbol_sizes


def get_julia_function(
    symbol,
    funcname="f",
    input_parameter_order=None,
    len_rhs=None,
    preallocate=True,
    round_constants=True,
):
    """
    Converts a pybamm expression tree into pure julia code that will calculate the
    result of calling `evaluate(t, y)` on the given expression tree.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to julia code
    funcname : str, optional
        The name to give to the function (default 'f')
    input_parameter_order : list, optional
        List of input parameter names. Defines the order in which the input parameters
        are extracted from 'p' in the julia function that is created
    len_rhs : int, optional
        The number of ODEs in the discretized differential equations. This also
        determines whether the model has any algebraic equations: if None (default),
        the model is assume to have no algebraic parts and ``julia_str`` is compatible
        with an ODE solver. If not None, ``julia_str`` is compatible with a DAE solver
    preallocate : bool, optional
        Whether to write the function in a way that preallocates memory for the output.
        Default is True, which is faster. Must be False for the function to be
        modelingtoolkitized.

    Returns
    -------
    julia_str : str
        String of julia code, to be evaluated by ``julia.Main.eval``

    """
    if len_rhs is None:
        typ = "ode"
    else:
        typ = "dae"
        # Take away dy from the differential states
        # we will return a function of the form
        # out[] = .. - dy[] for the differential states
        # out[] = .. for the algebraic states
        symbol_minus_dy = []
        end = 0
        for child in symbol.orphans:
            start = end
            end += child.size
            if end <= len_rhs:
                symbol_minus_dy.append(child - pybamm.StateVectorDot(slice(start, end)))
            else:
                symbol_minus_dy.append(child)
        symbol = pybamm.numpy_concatenation(*symbol_minus_dy)
    constants, var_symbols, var_symbol_sizes = to_julia(
        symbol, round_constants=round_constants
    )

    # extract constants in generated function
    const_and_cache_str = "cs = (\n"
    shorter_const_names = {}
    for i_const, (symbol_id, const_value) in enumerate(constants.items()):
        const_name = id_to_julia_variable(symbol_id, "const")
        const_name_short = "const_{}".format(i_const)
        const_and_cache_str += "   {} = {},\n".format(const_name_short, const_value)
        shorter_const_names[const_name] = const_name_short

    # Pop (get and remove) items from the dictionary of symbols one by one
    # If they are simple operations (@view, +, -, *, /), replace all future
    # occurences instead of assigning them. This "inlining" speeds up the computation
    inlineable_symbols = ["@view", "+", "-", "*", "/"]
    var_str = ""
    input_parameters = {}
    while var_symbols:
        var_symbol_id, symbol_line = var_symbols.popitem(last=False)
        julia_var = id_to_julia_variable(var_symbol_id, "cache")
        # Look for lists in the variable symbols. These correpsond to concatenations, so
        # assign the children to the right parts of the vector
        if symbol_line[0] == "[" and symbol_line[-1] == "]":
            # convert to actual list
            symbol_line = symbol_line[1:-1].split(", ")
            start = 0
            if preallocate is True or var_symbol_id == symbol.id:
                for child_size_and_name in symbol_line:
                    child_size, child_name = child_size_and_name.split("::")
                    end = start + int(child_size)
                    # add 1 to start to account for julia 1-indexing
                    var_str += "@. {}[{}:{}] = {}\n".format(
                        julia_var, start + 1, end, child_name
                    )
                    start = end
            else:
                concat_str = "{} = vcat(".format(julia_var)
                for i, child_size_and_name in enumerate(symbol_line):
                    child_size, child_name = child_size_and_name.split("::")
                    var_str += "x{} = @. {}\n".format(i + 1, child_name)
                    concat_str += "x{}, ".format(i + 1)
                var_str += concat_str[:-2] + ")\n"
        # use mul! for matrix multiplications (requires LinearAlgebra library)
        elif " @ " in symbol_line:
            if preallocate is False:
                symbol_line = symbol_line.replace(" @ ", " * ")
                var_str += "{} = {}\n".format(julia_var, symbol_line)
            else:
                symbol_line = symbol_line.replace(" @ ", ", ")
                var_str += "mul!({}, {})\n".format(julia_var, symbol_line)
        # find input parameters
        elif symbol_line.startswith("inputs"):
            input_parameters[julia_var] = symbol_line[8:-2]
        elif "minimum" in symbol_line or "maximum" in symbol_line:
            var_str += "{} .= {}\n".format(julia_var, symbol_line)
        else:
            # don't replace the matrix multiplication cases (which will be
            # turned into a mul!), since it is faster to assign to a cache array
            # first in that case
            # e.g. mul!(cs.cache_1, cs.cache_2, cs.cache_3)
            # unless it is a @view in which case we don't
            # need to cache
            # e.g. mul!(cs.cache_1, cs.cache_2, @view y[1:10])
            # also don't replace the minimum() or maximum() cases as we can't
            # broadcast them
            any_matmul_min_max = any(
                julia_var in next_symbol_line
                and (
                    any(
                        x in next_symbol_line
                        for x in [" @ ", "mul!", "minimum", "maximum"]
                    )
                    and not symbol_line.startswith("@view")
                )
                for next_symbol_line in var_symbols.values()
            )
            # inline operation if it can be inlined
            if (
                any(x in symbol_line for x in inlineable_symbols) or symbol_line == "t"
            ) and not any_matmul_min_max:
                found_replacement = False
                # replace all other occurrences of the variable
                # in the dictionary with the symbol line
                for next_var_id, next_symbol_line in var_symbols.items():
                    if julia_var in next_symbol_line:
                        if symbol_line == "t":
                            # no brackets needed
                            var_symbols[next_var_id] = next_symbol_line.replace(
                                julia_var, symbol_line
                            )
                        else:
                            # add brackets so that the order of operations is maintained
                            var_symbols[next_var_id] = next_symbol_line.replace(
                                julia_var, "({})".format(symbol_line)
                            )
                        found_replacement = True
                if not found_replacement:
                    var_str += "@. {} = {}\n".format(julia_var, symbol_line)

            # otherwise assign
            else:
                var_str += "@. {} = {}\n".format(julia_var, symbol_line)
    # Replace all input parameter names
    for input_parameter_id, input_parameter_name in input_parameters.items():
        var_str = var_str.replace(input_parameter_id, input_parameter_name)

    # indent code
    var_str = "   " + var_str
    var_str = var_str.replace("\n", "\n   ")

    # add the cache variables to the cache NamedTuple
    i_cache = 0
    for var_symbol_id, var_symbol_size in var_symbol_sizes.items():
        # Skip caching the result variable since this is provided as dy
        # Also skip caching the result variable if it doesn't appear in the var_str,
        # since it has been inlined and does not need to be assigned to
        julia_var = id_to_julia_variable(var_symbol_id, "cache")
        if var_symbol_id != symbol.id and julia_var in var_str:
            julia_var_short = "cache_{}".format(i_cache)
            var_str = var_str.replace(julia_var, julia_var_short)
            i_cache += 1
            if preallocate is True:
                const_and_cache_str += "   {} = zeros({}),\n".format(
                    julia_var_short, var_symbol_size
                )
            else:
                # Cache variables have not been preallocated
                var_str = var_str.replace(
                    "@. {} = ".format(julia_var_short),
                    "{} = @. ".format(julia_var_short),
                )

    # Shorten the name of the constants from id to const_0, const_1, etc.
    for long, short in shorter_const_names.items():
        var_str = var_str.replace(long, "cs." + short)

    # close the constants and cache string
    const_and_cache_str += ")\n"

    # remove the constant and cache sring if it is empty
    const_and_cache_str = const_and_cache_str.replace("cs = (\n)\n", "")

    # calculate the final variable that will output the result
    if symbol.is_constant():
        result_var = id_to_julia_variable(symbol.id, "const")
        if result_var in shorter_const_names:
            result_var = shorter_const_names[result_var]
        result_value = symbol.evaluate()
        if isinstance(result_value, numbers.Number):
            var_str = var_str + "\n   dy .= " + str(result_value) + "\n"
        else:
            var_str = var_str + "\n   dy .= cs." + result_var + "\n"
    else:
        result_var = id_to_julia_variable(symbol.id, "cache")
        if typ == "ode":
            out = "dy"
        elif typ == "dae":
            out = "out"
        # replace "cache_123 = ..." with "dy .= ..." (ensure we allocate to the
        # variable that was passed in)
        var_str = var_str.replace(f"   {result_var} =", f"   {out} .=")
        # catch other cases for dy
        var_str = var_str.replace(result_var, out)

    # add "cs." to cache names
    if preallocate is True:
        var_str = var_str.replace("cache", "cs.cache")

    # line that extracts the input parameters in the right order
    if input_parameter_order is None:
        input_parameter_extraction = ""
    elif len(input_parameter_order) == 1:
        # extract the single parameter
        input_parameter_extraction = "   " + input_parameter_order[0] + " = p[1]\n"
    else:
        # extract all parameters
        input_parameter_extraction = "   " + ", ".join(input_parameter_order) + " = p\n"

    if preallocate is False or const_and_cache_str == "":
        func_def = f"{funcname}!"
    else:
        func_def = f"{funcname}_with_consts!"

    # add function def
    if typ == "ode":
        function_def = f"\nfunction {func_def}(dy, y, p, t)\n"
    elif typ == "dae":
        function_def = f"\nfunction {func_def}(out, dy, y, p, t)\n"
    julia_str = (
        "begin\n"
        + const_and_cache_str
        + function_def
        + input_parameter_extraction
        + var_str
    )

    # close the function, with a 'nothing' to avoid allocations
    julia_str += "nothing\nend\n\n"
    julia_str = julia_str.replace("\n   \n", "\n")

    if not (preallocate is False or const_and_cache_str == ""):
        # Use a let block for the cached variables
        # open the let block
        julia_str = julia_str.replace("cs = (", f"{funcname}! = let cs = (")
        # close the let block
        julia_str += "end\n"

    # close the "begin"
    julia_str += "end"

    return julia_str


def convert_var_and_eqn_to_str(var, eqn, all_constants_str, all_variables_str, typ):
    """
    Converts a variable and its equation to a julia string

    Parameters
    ----------
    var : :class:`pybamm.Symbol`
        The variable (key in the dictionary of rhs/algebraic/initial conditions)
    eqn : :class:`pybamm.Symbol`
        The equation (value in the dictionary of rhs/algebraic/initial conditions)
    all_constants_str : str
        String containing all the constants defined so far
    all_variables_str : str
        String containing all the variables defined so far
    typ : str
        The type of the variable/equation pair being converted ("equation", "initial
        condition", or "boundary condition")

    Returns
    -------
    all_constants_str : str
        Updated string of all constants
    all_variables_str : str
        Updated string of all variables
    eqn_str : str
        The string describing the final equation result, perhaps as a function of some
        variables and/or constants in all_constants_str and all_variables_str

    """
    if isinstance(eqn, pybamm.Broadcast):
        # ignore broadcasts for now
        eqn = eqn.child

    var_symbols = to_julia(eqn)[1]

    # var_str = ""
    # for symbol_id, symbol_line in var_symbols.items():
    #     var_str += f"{id_to_julia_variable(symbol_id)} = {symbol_line}\n"
    # Pop (get and remove) items from the dictionary of symbols one by one
    # If they are simple operations (+, -, *, /), replace all future
    # occurences instead of assigning them.
    inlineable_symbols = [" + ", " - ", " * ", " / "]
    var_str = ""
    while var_symbols:
        var_symbol_id, symbol_line = var_symbols.popitem(last=False)
        julia_var = id_to_julia_variable(var_symbol_id, "cache")
        # inline operation if it can be inlined
        if "concatenation" not in symbol_line:
            found_replacement = False
            # replace all other occurrences of the variable
            # in the dictionary with the symbol line
            for next_var_id, next_symbol_line in var_symbols.items():
                if (
                    symbol_line == "t"
                    or " " not in symbol_line
                    or symbol_line.startswith("grad")
                    or not any(x in next_symbol_line for x in inlineable_symbols)
                ):
                    # cases that don't need brackets
                    var_symbols[next_var_id] = next_symbol_line.replace(
                        julia_var, symbol_line
                    )
                elif next_symbol_line.startswith("concatenation"):
                    var_symbols[next_var_id] = next_symbol_line.replace(
                        julia_var, f"\n   {symbol_line}\n"
                    )
                else:
                    # add brackets so that the order of operations is maintained
                    var_symbols[next_var_id] = next_symbol_line.replace(
                        julia_var, "({})".format(symbol_line)
                    )
                found_replacement = True
            if not found_replacement:
                var_str += "{} = {}\n".format(julia_var, symbol_line)

        # otherwise assign
        else:
            var_str += "{} = {}\n".format(julia_var, symbol_line)

    # If we have created a concatenation we need to define it
    # Hardcoded to the negative electrode, separator, positive electrode case for now
    if "concatenation" in var_str and "function concatenation" not in all_variables_str:
        concatenation_def = (
            "\nfunction concatenation(n, s, p)\n"
            + "   # A concatenation in the electrolyte domain\n"
            + "   IfElse.ifelse(\n"
            + "      x < neg_width, n, IfElse.ifelse(\n"
            + "         x < neg_plus_sep_width, s, p\n"
            + "      )\n"
            + "   )\n"
            + "end\n"
        )
    else:
        concatenation_def = ""

    # Define the FunctionParameter objects that have not yet been defined
    function_defs = ""
    for x in eqn.pre_order():
        if (
            isinstance(x, pybamm.FunctionParameter)
            and f"function {x.name}" not in all_variables_str
            and typ == "equation"
        ):
            function_def = (
                f"\nfunction {x.name}("
                + ", ".join(x.arg_names)
                + ")\n"
                + "   {}\n".format(str(x.callable).replace("**", "^"))
                + "end\n"
            )
            function_defs += function_def

    if concatenation_def + function_defs != "":
        function_defs += "\n"

    var_str = concatenation_def + function_defs + var_str

    # add a comment labeling the equation, and the equation itself
    if var_str == "":
        all_variables_str += ""
    else:
        all_variables_str += f"# '{var.name}' {typ}\n" + var_str + "\n"

    # calculate the final variable that will output the result
    if eqn.is_constant():
        result_var = id_to_julia_variable(eqn.id, "const")
    else:
        result_var = id_to_julia_variable(eqn.id, "cache")
    if is_constant_and_can_evaluate(eqn):
        result_value = eqn.evaluate()
    else:
        result_value = None

    # define the variable that goes into the equation
    if eqn.is_constant() and isinstance(result_value, numbers.Number):
        eqn_str = str(result_value)
    else:
        eqn_str = result_var

    return all_constants_str, all_variables_str, eqn_str


def get_julia_mtk_model(model, geometry=None, tspan=None):
    """
    Converts a pybamm model into a Julia ModelingToolkit model

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be converted
    geometry : dict, optional
        Dictionary defining the geometry. Must be provided if the model is a PDE model
    tspan : array-like, optional
        Time for which to solve the model. Must be provided if the model is a PDE model

    Returns
    -------
    mtk_str : str
        String of julia code representing a model in MTK,
        to be evaluated by ``julia.Main.eval``
    """
    # Extract variables
    variables = {**model.rhs, **model.algebraic}.keys()
    variable_to_print_name = {}
    for i, var in enumerate(variables):
        if var.print_name is not None:
            print_name = var._raw_print_name
        else:
            print_name = f"u{i+1}"
        variable_to_print_name[var] = print_name
        if isinstance(var, pybamm.ConcatenationVariable):
            for child in var.children:
                variable_to_print_name[child] = print_name

    # Extract domain and auxiliary domains
    all_domains = set(
        [tuple(dom) for var in variables for dom in var.domains.values() if dom != []]
    )
    is_pde = bool(all_domains)

    # Check geometry and tspan have been provided if a PDE
    if is_pde:
        if geometry is None:
            raise ValueError("must provide geometry if the model is a PDE model")
        if tspan is None:
            raise ValueError("must provide tspan if the model is a PDE model")

    # Read domain names
    domain_name_to_symbol = {}
    long_domain_symbol_to_short = {}
    for dom in all_domains:
        # Read domain name from geometry
        domain_symbol = list(geometry[dom[0]].keys())[0]
        if len(dom) > 1:
            domain_symbol = domain_symbol[0]
            # For multi-domain variables keep only the first letter of the domain
            domain_name_to_symbol[tuple(dom)] = domain_symbol
            # Record which domain symbols we shortened
            for d in dom:
                long = list(geometry[d].keys())[0]
                long_domain_symbol_to_short[long] = domain_symbol
        else:
            # Otherwise keep the whole domain
            domain_name_to_symbol[tuple(dom)] = domain_symbol

    # Read domain limits
    domain_name_to_limits = {(): None}
    for dom in all_domains:
        limits = list(geometry[dom[0]].values())[0].values()
        if len(limits) > 1:
            lower_limit, _ = list(geometry[dom[0]].values())[0].values()
            _, upper_limit = list(geometry[dom[-1]].values())[0].values()
            domain_name_to_limits[tuple(dom)] = (
                lower_limit.evaluate(),
                upper_limit.evaluate(),
            )
        else:
            # Don't record limits for variables that have "limits" of length 1 i.e.
            # a zero-dimensional domain
            domain_name_to_limits[tuple(dom)] = None

    # Define independent variables for each variable
    var_to_ind_vars = {}
    var_to_ind_vars_left_boundary = {}
    var_to_ind_vars_right_boundary = {}
    for var in variables:
        if var.domain in [[], ["current collector"]]:
            var_to_ind_vars[var] = "(t)"
        else:
            # all independent variables e.g. (t, x) or (t, rn, xn)
            domain_symbols = ", ".join(
                domain_name_to_symbol[tuple(dom)]
                for dom in var.domains.values()
                if domain_name_to_limits[tuple(dom)] is not None
            )
            var_to_ind_vars[var] = f"(t, {domain_symbols})"
            if isinstance(var, pybamm.ConcatenationVariable):
                for child in var.children:
                    var_to_ind_vars[child] = f"(t, {domain_symbols})"
            aux_domain_symbols = ", ".join(
                domain_name_to_symbol[tuple(dom)]
                for level, dom in var.domains.items()
                if level != "primary" and domain_name_to_limits[tuple(dom)] is not None
            )
            if aux_domain_symbols != "":
                aux_domain_symbols = ", " + aux_domain_symbols

            limits = domain_name_to_limits[tuple(var.domain)]
            # left bc e.g. (t, 0) or (t, 0, xn)
            var_to_ind_vars_left_boundary[var] = f"(t, {limits[0]}{aux_domain_symbols})"
            # right bc e.g. (t, 1) or (t, 1, xn)
            var_to_ind_vars_right_boundary[
                var
            ] = f"(t, {limits[1]}{aux_domain_symbols})"

    mtk_str = "begin\n"
    # Define parameters (including independent variables)
    # Makes a line of the form '@parameters t x1 x2 x3 a b c d'
    ind_vars = ["t"] + [
        sym
        for dom, sym in domain_name_to_symbol.items()
        if domain_name_to_limits[dom] is not None
    ]
    for domain_name, domain_symbol in domain_name_to_symbol.items():
        if domain_name_to_limits[domain_name] is not None:
            mtk_str += f"# {domain_name} -> {domain_symbol}\n"
    mtk_str += "@parameters " + " ".join(ind_vars)
    if len(model.input_parameters) > 0:
        mtk_str += "\n# Input parameters\n@parameters"
        for param in model.input_parameters:
            mtk_str += f" {param.name}"
    mtk_str += "\n"

    # Add a comment with the variable names
    for var in variables:
        mtk_str += f"# '{var.name}' -> {variable_to_print_name[var]}\n"
    # Makes a line of the form '@variables u1(t) u2(t)'
    dep_vars = []
    mtk_str += "@variables"
    for var in variables:
        mtk_str += f" {variable_to_print_name[var]}(..)"
        dep_var = variable_to_print_name[var] + var_to_ind_vars[var]
        dep_vars.append(dep_var)
    mtk_str += "\n"

    # Define derivatives
    for domain_symbol in ind_vars:
        mtk_str += f"D{domain_symbol} = Differential({domain_symbol})\n"
    mtk_str += "\n"

    # Define equations
    all_eqns_str = ""
    all_constants_str = ""
    all_julia_str = ""
    for var, eqn in {**model.rhs, **model.algebraic}.items():
        all_constants_str, all_julia_str, eqn_str = convert_var_and_eqn_to_str(
            var, eqn, all_constants_str, all_julia_str, "equation"
        )

        if var in model.rhs:
            all_eqns_str += (
                f"   Dt({variable_to_print_name[var]}{var_to_ind_vars[var]}) "
                + f"~ {eqn_str},\n"
            )
        elif var in model.algebraic:
            all_eqns_str += f"   0 ~ {eqn_str},\n"

    # Replace any long domain symbols with the short version
    # e.g. "xn" gets replaced with "x"
    for long, short in long_domain_symbol_to_short.items():
        # we need to add a space to avoid accidentally replacing 'exp' with 'ex'
        all_julia_str = all_julia_str.replace(" " + long, " " + short)

    # Replace variables in the julia strings that correspond to pybamm variables with
    # their julia equivalent
    for var, julia_id in variable_to_print_name.items():
        # e.g. boundary_value_right(cache_123456789) gets replaced with u1(t, 1)
        cache_var_id = id_to_julia_variable(var.id, "cache")
        if f"boundary_value_right({cache_var_id})" in all_julia_str:
            all_julia_str = all_julia_str.replace(
                f"boundary_value_right({cache_var_id})",
                julia_id + var_to_ind_vars_right_boundary[var],
            )
        # e.g. cache_123456789 gets replaced with u1(t, x)
        all_julia_str = all_julia_str.replace(
            cache_var_id, julia_id + var_to_ind_vars[var]
        )

    # Replace independent variables (domain names) in julia strings with the
    # corresponding symbol
    for domain_name, domain_symbol in domain_name_to_symbol.items():
        all_julia_str = all_julia_str.replace(
            f"grad_{domain_name}", f"D{domain_symbol}"
        )
        # Different divergence depending on the coordinate system
        coord_sys = getattr(pybamm.standard_spatial_vars, domain_symbol).coord_sys
        if coord_sys == "cartesian":
            all_julia_str = all_julia_str.replace(
                f"div_{domain_name}", f"D{domain_symbol}"
            )
        elif coord_sys == "spherical polar":
            all_julia_str = all_julia_str.replace(
                f"div_{domain_name}(",
                f"1 / {domain_symbol}^2 * D{domain_symbol}({domain_symbol}^2 * ",
            )

    # Replace the thicknesses in the concatenation with the actual thickness from the
    # geometry
    if "neg_width" in all_julia_str or "neg_plus_sep_width" in all_julia_str:
        var = pybamm.standard_spatial_vars
        x_n = geometry["negative electrode"]["x_n"]["max"].evaluate()
        x_s = geometry["separator"]["x_s"]["max"].evaluate()
        all_julia_str = all_julia_str.replace("neg_width", str(x_n))
        all_julia_str = all_julia_str.replace("neg_plus_sep_width", str(x_s))

    # Update the MTK string
    mtk_str += all_constants_str + all_julia_str + "\n" + f"eqs = [\n{all_eqns_str}]\n"

    ####################################################################################
    # Initial and boundary conditions
    ####################################################################################
    # Initial conditions
    all_ic_bc_str = "   # initial conditions\n"
    all_ic_bc_constants_str = ""
    all_ic_bc_julia_str = ""
    for var, eqn in model.initial_conditions.items():
        (
            all_ic_bc_constants_str,
            all_ic_bc_julia_str,
            eqn_str,
        ) = convert_var_and_eqn_to_str(
            var, eqn, all_ic_bc_constants_str, all_ic_bc_julia_str, "initial condition"
        )

        if not is_pde:
            all_ic_bc_str += f"   {variable_to_print_name[var]}(t) => {eqn_str},\n"
        else:
            if var.domain == []:
                doms = ""
            else:
                doms = ", " + domain_name_to_symbol[tuple(var.domain)]

            all_ic_bc_str += f"   {variable_to_print_name[var]}(0{doms}) ~ {eqn_str},\n"
    # Boundary conditions
    if is_pde:
        all_ic_bc_str += "   # boundary conditions\n"
        for var, eqn_side in model.boundary_conditions.items():
            if isinstance(var, (pybamm.Variable, pybamm.ConcatenationVariable)):
                for side, (eqn, typ) in eqn_side.items():
                    (
                        all_ic_bc_constants_str,
                        all_ic_bc_julia_str,
                        eqn_str,
                    ) = convert_var_and_eqn_to_str(
                        var,
                        eqn,
                        all_ic_bc_constants_str,
                        all_ic_bc_julia_str,
                        "boundary condition",
                    )

                    if side == "left":
                        limit = var_to_ind_vars_left_boundary[var]
                    elif side == "right":
                        limit = var_to_ind_vars_right_boundary[var]

                    bc = f"{variable_to_print_name[var]}{limit}"
                    if typ == "Dirichlet":
                        bc = bc
                    elif typ == "Neumann":
                        bc = f"D{domain_name_to_symbol[tuple(var.domain)]}({bc})"
                    all_ic_bc_str += f"   {bc} ~ {eqn_str},\n"

    # Replace variables in the julia strings that correspond to pybamm variables with
    # their julia equivalent
    for var, julia_id in variable_to_print_name.items():
        # e.g. boundary_value_right(cache_123456789) gets replaced with u1(t, 1)
        cache_var_id = id_to_julia_variable(var.id, "cache")
        if f"boundary_value_right({cache_var_id})" in all_ic_bc_julia_str:
            all_ic_bc_julia_str = all_ic_bc_julia_str.replace(
                f"boundary_value_right({cache_var_id})",
                julia_id + var_to_ind_vars_right_boundary[var],
            )
        # e.g. cache_123456789 gets replaced with u1(t, x)
        all_ic_bc_julia_str = all_ic_bc_julia_str.replace(
            cache_var_id, julia_id + var_to_ind_vars[var]
        )

    ####################################################################################

    # Create ODESystem or PDESystem
    if not is_pde:
        mtk_str += "sys = ODESystem(eqs, t)\n\n"
        mtk_str += (
            all_ic_bc_constants_str
            + all_ic_bc_julia_str
            + "\n"
            + f"u0 = [\n{all_ic_bc_str}]\n"
        )
    else:
        # Initial and boundary conditions
        mtk_str += (
            all_ic_bc_constants_str
            + all_ic_bc_julia_str
            + "\n"
            + f"ics_bcs = [\n{all_ic_bc_str}]\n"
        )

        # Domains
        mtk_str += "\n"
        tpsan_str = ",".join(
            map(lambda x: f"{x / model.timescale.evaluate():.3f}", tspan)
        )
        mtk_str += f"t_domain = Interval({tpsan_str})\n"
        domains = "domains = [\n   t in t_domain,\n"
        for domain, symbol in domain_name_to_symbol.items():
            limits = domain_name_to_limits[tuple(domain)]
            if limits is not None:
                mtk_str += f"{symbol}_domain = Interval{limits}\n"
                domains += f"   {symbol} in {symbol}_domain,\n"
        domains += "]\n"

        mtk_str += "\n"
        mtk_str += domains

        # Independent and dependent variables
        mtk_str += "ind_vars = [{}]\n".format(", ".join(ind_vars))
        mtk_str += "dep_vars = [{}]\n\n".format(", ".join(dep_vars))

        name = model.name.replace(" ", "_").replace("-", "_")
        mtk_str += (
            name
            + "_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)\n\n"
        )

    # Replace parameters in the julia strings in the form "inputs[name]"
    # with just "name"
    for param in model.input_parameters:
        mtk_str = mtk_str.replace(f"inputs['{param.name}']", param.name)

    # Need to add 'nothing' to the end of the mtk string to avoid errors in MTK v4
    # See https://github.com/SciML/diffeqpy/issues/82
    mtk_str += "nothing\nend\n"

    return mtk_str
