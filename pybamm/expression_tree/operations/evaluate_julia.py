#
# Write a symbol to Julia
#
import pybamm

import numpy as np
import scipy.sparse
import re
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
        var_format = "cache_{:05d}"

    # Need to replace "-" character to make them valid julia variable names
    return var_format.format(symbol_id).replace("-", "m")


def find_symbols(symbol, constant_symbols, variable_symbols, variable_symbol_sizes):
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
    if symbol.is_constant():
        value = symbol.evaluate()
        if not isinstance(value, numbers.Number):
            if scipy.sparse.issparse(value):
                # Create Julia SparseArray
                row, col, data = scipy.sparse.find(value)
                m, n = value.shape
                # Set print options large enough to avoid ellipsis
                # at least as big is len(row) = len(col) = len(data)
                np.set_printoptions(
                    threshold=max(np.get_printoptions()["threshold"], len(row) + 10)
                )
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
        find_symbols(child, constant_symbols, variable_symbols, variable_symbol_sizes)

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
        if isinstance(symbol, pybamm.MatrixMultiplication):
            symbol_str = "{0} * {1}".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Inner):
            symbol_str = "{0} .* {1}".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Minimum):
            symbol_str = "min.({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Maximum):
            symbol_str = "max.({},{})".format(children_vars[0], children_vars[1])
        elif isinstance(symbol, pybamm.Power):
            # julia uses ^ instead of ** for power
            # include dot for elementwise operations
            symbol_str = children_vars[0] + " .^ " + children_vars[1]
        else:
            # all other operations use the same symbol
            # include dot: all other operations should be elementwise
            symbol_str = children_vars[0] + " ." + symbol.name + " " + children_vars[1]

    elif isinstance(symbol, pybamm.UnaryOperator):
        # Index has a different syntax than other univariate operations
        if isinstance(symbol, pybamm.Index):
            # Because of how julia indexing works, add 1 to the start, but not to the
            # stop
            symbol_str = "{}[{}:{}]".format(
                children_vars[0], symbol.slice.start + 1, symbol.slice.stop
            )
        elif isinstance(symbol, (pybamm.Gradient, pybamm.Divergence)):
            symbol_str = "D'{}'({})".format(symbol.domain[0], children_vars[0])
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
        if isinstance(symbol, (pybamm.Min, pybamm.Max)):
            symbol_str = "{}({})".format(julia_name, children_str)
        else:
            symbol_str = "{}.({})".format(julia_name, children_str)

    elif isinstance(symbol, pybamm.Concatenation):

        # don't bother to concatenate if there is only a single child
        if isinstance(symbol, (pybamm.NumpyConcatenation, pybamm.SparseStack)):
            if len(children_vars) == 1:
                symbol_str = children_vars[0]
            else:
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
            if len(children_vars) > 1 or symbol.secondary_dimensions_npts > 1:
                # return a list of the children variables, which will be converted to a
                # line by line assignment
                # return this as a string so that other functionality still works
                # also save sizes
                symbol_str = "["
                for child, size in zip(all_child_vectors, all_child_sizes):
                    symbol_str += "{}::{}, ".format(size, child)
                symbol_str = symbol_str[:-2] + "]"
            else:
                raise NotImplementedError

    # Note: we assume that y is being passed as a column vector
    elif isinstance(symbol, pybamm.StateVector):
        indices = np.argwhere(symbol.evaluation_array).reshape(-1).astype(np.int32)
        # add 1 since julia uses 1-indexing
        indices += 1
        consecutive = np.all(indices[1:] - indices[:-1] == 1)
        if len(indices) == 1:
            symbol_str = "@view y[{}]".format(indices[0])
        elif consecutive:
            # julia does include the final value
            symbol_str = "@view y[{}:{}]".format(indices[0], indices[-1])
        else:
            indices_array = pybamm.Array(indices)
            # Save the indices as constant by printing to a string
            # Set print options large enough to avoid ellipsis
            # at least as big as len(row) = len(col) = len(data)
            np.set_printoptions(
                threshold=max(np.get_printoptions()["threshold"], indices.shape[0] + 10)
            )
            constant_symbols[indices_array.id] = np.array2string(indices, separator=",")
            index_name = id_to_julia_variable(indices_array.id, True)
            symbol_str = "@view y[{}]".format(index_name)

    elif isinstance(symbol, pybamm.Time):
        symbol_str = "t"

    elif isinstance(symbol, pybamm.InputParameter):
        symbol_str = "inputs['{}']".format(symbol.name)

    elif isinstance(symbol, pybamm.Variable):
        # No need to do anything if a Variable is found
        return

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
        elif symbol.shape[1] == 1:
            variable_symbol_sizes[symbol.id] = symbol.shape[0]
        else:
            raise ValueError("expected scalar or column vector")
    except NotImplementedError:
        pass


def to_julia(symbol):
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
    find_symbols(symbol, constant_values, variable_symbols, variable_symbol_sizes)

    return constant_values, variable_symbols, variable_symbol_sizes


def get_julia_function(symbol, funcname="f"):
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

    constants, var_symbols, var_symbol_sizes = to_julia(symbol)

    # extract constants in generated function
    const_and_cache_str = "cs = (\n"
    for symbol_id, const_value in constants.items():
        const_name = id_to_julia_variable(symbol_id, True)
        const_and_cache_str += "   {} = {},\n".format(const_name, const_value)

    # Pop (get and remove) items from the dictionary of symbols one by one
    # If they are simple operations (@view, .+, .-, .*, ./), replace all future
    # occurences instead of assigning them. This "inlining" speeds up the computation
    inlineable_symbols = ["@view", ".+", ".-", ".*", "./"]
    var_str = ""
    while var_symbols:
        var_symbol_id, symbol_line = var_symbols.popitem(last=False)
        julia_var = id_to_julia_variable(var_symbol_id, False)
        # Look for lists in the variable symbols. These correpsond to concatenations, so
        # assign the children to the right parts of the vector
        if symbol_line[0] == "[" and symbol_line[-1] == "]":
            # convert to actual list
            symbol_line = symbol_line[1:-1].split(", ")
            start = 0
            for child_size_and_name in symbol_line:
                child_size, child_name = child_size_and_name.split("::")
                end = start + int(child_size)
                # add 1 to start to account for julia 1-indexing
                var_str += "{}[{}:{}] .= {}\n".format(
                    julia_var, start + 1, end, child_name
                )
                start = end
        # use mul! for matrix multiplications (requires LinearAlgebra library)
        elif " * " in symbol_line:
            symbol_line = symbol_line.replace(" * ", ", ")
            var_str += "mul!({}, {})\n".format(julia_var, symbol_line)
        else:
            # inline operation if it can be inlined
            if any(x in symbol_line for x in inlineable_symbols) or symbol_line == "t":
                found_replacement = False
                # replace all other occurrences of the variable
                # in the dictionary with the symbol line
                for next_var_id, next_symbol_line in var_symbols.items():
                    # don't replace the matrix multiplication cases (which will be
                    # turned into a mul!), since it is faster to assign to a cache array
                    # first in that case, unless it is a @view in which case we don't
                    # need to cache
                    if julia_var in next_symbol_line and not (
                        (" * " in next_symbol_line or "mul!" in next_symbol_line)
                        and not symbol_line.startswith("@view")
                    ):
                        if symbol_line != "t":
                            # add brackets so that the order of operations is maintained
                            var_symbols[next_var_id] = next_symbol_line.replace(
                                julia_var, "({})".format(symbol_line)
                            )
                        else:
                            # add brackets so that the order of operations is maintained
                            var_symbols[next_var_id] = next_symbol_line.replace(
                                julia_var, symbol_line
                            )
                        found_replacement = True
                if not found_replacement:
                    var_str += "{} .= {}\n".format(julia_var, symbol_line)

            # otherwise assign
            else:
                var_str += "{} .= {}\n".format(julia_var, symbol_line)
    # add "cs." to constant and cache names
    var_str = var_str.replace("const", "cs.const")
    var_str = var_str.replace("cache", "cs.cache")
    # indent code
    var_str = "   " + var_str
    var_str = var_str.replace("\n", "\n   ")

    # add the cache variables to the cache NamedTuple
    for var_symbol_id, var_symbol_size in var_symbol_sizes.items():
        # Skip caching the result variable since this is provided as dy
        # Also skip caching the result variable if it doesn't appear in the var_str,
        # since it has been inlined and does not need to be assigned to
        julia_var = id_to_julia_variable(var_symbol_id, False)
        if var_symbol_id != symbol.id and julia_var in var_str:
            cache_name = id_to_julia_variable(var_symbol_id, False)
            const_and_cache_str += "   {} = zeros({}),\n".format(
                cache_name, var_symbol_size
            )

    # close the constants and cache string
    const_and_cache_str += ")\n"

    # remove the constant and cache sring if it is empty
    const_and_cache_str = const_and_cache_str.replace("cs = (\n)\n", "")

    # add function def and sparse arrays to first line
    imports = "begin\nusing SparseArrays, LinearAlgebra\n\n"
    julia_str = (
        imports
        + const_and_cache_str
        + f"\nfunction {funcname}_with_consts(dy, y, p, t)\n"
        + var_str
    )

    # calculate the final variable that will output the result
    result_var = id_to_julia_variable(symbol.id, symbol.is_constant())
    if symbol.is_constant():
        result_value = symbol.evaluate()

    # assign the return variable
    if symbol.is_constant():
        if isinstance(result_value, numbers.Number):
            julia_str = julia_str + "\n   dy .= " + str(result_value) + "\n"
        else:
            julia_str = julia_str + "\n   dy .= cs." + result_var + "\n"
    else:
        julia_str = julia_str.replace("cs." + result_var, "dy")

    # close the function
    julia_str += "end\n\n"
    julia_str = julia_str.replace("\n   end", "\nend")
    julia_str = julia_str.replace("\n   \n", "\n")

    if const_and_cache_str == "":
        julia_str += f"{funcname} = {funcname}_with_consts\n"
    else:
        # Use a let block for the cached variables
        # open the let block
        julia_str = julia_str.replace("cs = (", f"{funcname} = let cs = (")
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
    constants, variable_symbols = to_julia(eqn)[:2]
    line_format = "{} .= {}"

    variables_str = "\n".join(
        [
            f"{id_to_julia_variable(symbol_id)} = {symbol_line}"
            for symbol_id, symbol_line in variable_symbols.items()
        ]
    )

    # extract constants in generated function
    for eqn_id, const_value in constants.items():
        const_name = id_to_julia_variable(eqn_id, True)
        all_constants_str += "{} = {}\n".format(const_name, const_value)
        # TODO: avoid repeated constants definitions

    # add a comment labeling the equation, and the equation itself
    if variables_str == "":
        all_variables_str += ""
    else:
        all_variables_str += f"# '{var.name}' {typ}\n" + variables_str + "\n"

    # calculate the final variable that will output the result
    result_var = id_to_julia_variable(eqn.id, eqn.is_constant())
    if eqn.is_constant():
        result_value = eqn.evaluate()

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
    variable_id_to_number = {var.id: f"u{i+1}" for i, var in enumerate(variables)}
    all_domains = list(set([dom for var in variables for dom in var.domain]))

    is_pde = bool(all_domains)

    # Check geometry and tspan have been provided if a PDE
    if is_pde:
        if geometry is None:
            raise ValueError("must provide geometry if the model is a PDE model")
        if tspan is None:
            raise ValueError("must provide tspan if the model is a PDE model")

    domain_name_to_symbol = {
        dom: list(geometry[dom].keys())[0].name for i, dom in enumerate(all_domains)
    }

    mtk_str = "begin\n"
    # Define parameters (including independent variables)
    # Makes a line of the form '@parameters t x1 x2 x3 a b c d'
    ind_vars = ["t"] + list(domain_name_to_symbol.values())
    for dom, number in domain_name_to_symbol.items():
        mtk_str += f"# '{dom}' -> {number}\n"
    mtk_str += "@parameters " + " ".join(ind_vars)
    for param in model.input_parameters:
        mtk_str += f" {param.name}"
    mtk_str += "\n"

    # Add a comment with the variable names
    for var in variables:
        mtk_str += f"# '{var.name}' -> {variable_id_to_number[var.id]}\n"
    # Makes a line of the form '@variables u1(t) u2(t)'
    dep_vars = list(variable_id_to_number.values())
    mtk_str += "@variables"
    var_to_ind_vars = {}
    for var in variables:
        if var.domain == []:
            var_to_ind_vars[var.id] = "(t)"
        else:
            var_to_ind_vars[var.id] = (
                "(t, "
                + ", ".join([domain_name_to_symbol[dom] for dom in var.domain])
                + ")"
            )
        mtk_str += f" {variable_id_to_number[var.id]}(..)"
    mtk_str += "\n"

    # Define derivatives

    mtk_str += "@derivatives Dt'~t\n"
    if is_pde:
        mtk_str += "@derivatives "
        for domain_symbol in domain_name_to_symbol.values():
            mtk_str += f"D{domain_symbol}'~{domain_symbol}"
        mtk_str += "\n"
    mtk_str += "\n"

    # Define equations
    all_eqns_str = ""
    all_constants_str = ""
    all_julia_str = ""
    for var, eqn in {**model.rhs, **model.algebraic}.items():
        all_constants_str, all_julia_str, eqn_str = convert_var_and_eqn_to_str(
            var, eqn, all_constants_str, all_julia_str, "equation"
        )
        # add

        if var in model.rhs:
            all_eqns_str += (
                f"   Dt({variable_id_to_number[var.id]}{var_to_ind_vars[var.id]}) "
                + f"~ {eqn_str},\n"
            )
        elif var in model.algebraic:
            all_eqns_str += f"   0 ~ {eqn_str},\n"

    # Replace variables in the julia strings that correspond to pybamm variables with
    # their julia equivalent
    # e.g. cache_123456789 gets replaced with u1(t, x)
    for var_id, julia_id in variable_id_to_number.items():
        all_julia_str = all_julia_str.replace(
            id_to_julia_variable(var_id, False), julia_id + var_to_ind_vars[var_id]
        )

    # Replace independent variables (domain names) in julia strings with the
    # corresponding symbol
    for domain, symbol in domain_name_to_symbol.items():
        all_julia_str = all_julia_str.replace(f"'{domain}'", symbol)

    # Replace parameters in the julia strings in the form "inputs[name]"
    # with just "name"
    for param in model.input_parameters:
        # Replace 'var_id' with 'param.name'
        all_julia_str = all_julia_str.replace(
            id_to_julia_variable(param.id, False), param.name
        )
        # Remove the line where the variable is re-defined
        all_julia_str = all_julia_str.replace(
            f"{param.name} = inputs['{param.name}']\n", ""
        )

    # Update the MTK string
    mtk_str += all_constants_str + all_julia_str + "\n" + f"eqs = [\n{all_eqns_str}]\n"

    ####################################################################################
    # Initial and boundary conditions
    ####################################################################################
    # Initial conditions
    all_ic_bc_str = ""
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
            all_ic_bc_str += f"   {variable_id_to_number[var.id]}(t) => {eqn_str},\n"
        else:
            doms = ", ".join([domain_name_to_symbol[dom] for dom in var.domain])
            all_ic_bc_str += (
                f"   {variable_id_to_number[var.id]}(0, {doms}) ~ {eqn_str},\n"
            )
    # Boundary conditions
    if is_pde:
        for var, eqn_side in model.boundary_conditions.items():
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

                geom = list(geometry[var.domain[0]].values())[0]
                if side == "left":
                    limit = geom["min"]
                elif side == "right":
                    limit = geom["max"]
                if typ == "Dirichlet":
                    pass
                elif typ == "Neumann":
                    raise NotImplementedError
                all_ic_bc_str += (
                    f"   {variable_id_to_number[var.id]}(t, {limit}) ~ {eqn_str},\n"
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
        mtk_str += f"t_domain = IntervalDomain({tspan[0]}, {tspan[1]})\n"
        domains = f"domains = [\n   t in t_domain,\n"
        for domain, symbol in domain_name_to_symbol.items():
            dom_limits = list(geometry[domain].values())[0]
            dom_min, dom_max = dom_limits.values()
            mtk_str += f"{symbol}_domain = IntervalDomain({dom_min}, {dom_max})\n"
            domains += f"   {symbol} in {symbol}_domain,\n"
        domains += "]\n"

        mtk_str += "\n"
        mtk_str += domains

        # Independent and dependent variables
        mtk_str += "ind_vars = [{}]\n".format(", ".join(ind_vars))
        mtk_str += "dep_vars = [{}]\n\n".format(", ".join(dep_vars))

        mtk_str += (
            "pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)\n\n"
        )

    # Need to add 'nothing' to the end of the mtk string to avoid errors in MTK v4
    # See https://github.com/SciML/diffeqpy/issues/82
    mtk_str += "nothing\nend\n"

    return mtk_str
