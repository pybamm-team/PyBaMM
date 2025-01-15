#
# Write a symbol to python
#
from __future__ import annotations
import numbers
from collections import OrderedDict
from numpy.typing import ArrayLike

import numpy as np
import scipy.sparse

import pybamm

if pybamm.has_jax():
    import jax

    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    if platform != "metal":
        jax.config.update("jax_enable_x64", True)


class JaxCooMatrix:
    """
    A sparse matrix in COO format, with internal arrays using jax device arrays

    This matrix only has two operations supported, a multiply with a scalar, and a
    dot product with a dense vector. It can also be converted to a dense 2D jax
    device array

    Parameters
    ----------

    row: arraylike
        1D array holding row indices of non-zero entries
    col: arraylike
        1D array holding col indices of non-zero entries
    data: arraylike
        1D array holding non-zero entries
    shape: 2-element tuple (x, y)
        where x is the number of rows, and y the number of columns of the matrix
    """

    def __init__(
        self, row: ArrayLike, col: ArrayLike, data: ArrayLike, shape: tuple[int, int]
    ):
        if not pybamm.has_jax():  # pragma: no cover
            raise ModuleNotFoundError(
                "Jax or jaxlib is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver"
            )

        self.row = jax.numpy.array(row)
        self.col = jax.numpy.array(col)
        self.data = jax.numpy.array(data)
        self.shape = shape
        self.nnz = len(self.data)

    def toarray(self):
        """convert sparse matrix to a dense 2D array"""
        result = jax.numpy.zeros(self.shape, dtype=self.data.dtype)
        return result.at[self.row, self.col].add(self.data)

    def dot_product(self, b):
        """
        dot product of matrix with a dense column vector b

        Parameters
        ----------
        b: jax device array
            must have shape (n, 1)
        """
        # assume b is a column vector
        result = jax.numpy.zeros((self.shape[0], 1), dtype=b.dtype)
        return result.at[self.row].add(self.data.reshape(-1, 1) * b[self.col])

    def scalar_multiply(self, b: float):
        """
        multiply of matrix with a scalar b

        Parameters
        ----------
        b: Number or 1 element jax device array
            scalar value to multiply
        """
        # assume b is a scalar or ndarray with 1 element
        return JaxCooMatrix(self.row, self.col, (self.data * b).reshape(-1), self.shape)

    def multiply(self, b):
        """
        general matrix multiply not supported
        """
        raise NotImplementedError

    def __matmul__(self, b):
        """see self.dot_product"""
        return self.dot_product(b)


def create_jax_coo_matrix(value: scipy.sparse):
    """
    Creates a JaxCooMatrix from a scipy.sparse matrix

    Parameters
    ----------

    value: scipy.sparse matrix
        the sparse matrix to be converted
    """
    scipy_coo = value.tocoo()
    row = jax.numpy.asarray(scipy_coo.row)
    col = jax.numpy.asarray(scipy_coo.col)
    data = jax.numpy.asarray(scipy_coo.data)
    return JaxCooMatrix(row, col, data, value.shape)


def id_to_python_variable(symbol_id, constant=False):
    """
    This function defines the format for the python variable names used in find_symbols
    and to_python. Variable names are based on a nodes' id to make them unique
    """

    if constant:
        var_format = "const_{:05d}"
    else:
        var_format = "var_{:05d}"

    # Need to replace "-" character to make them valid python variable names
    return var_format.format(symbol_id).replace("-", "m")


def is_scalar(arg):
    is_number = isinstance(arg, numbers.Number)
    if is_number:
        return True
    else:
        return np.all(np.array(arg.shape) == 1)


def find_symbols(
    symbol: pybamm.Symbol,
    constant_symbols: OrderedDict,
    variable_symbols: OrderedDict,
    output_jax=False,
):
    """
    This function converts an expression tree to a dictionary of node id's and strings
    specifying valid python code to calculate that nodes value, given y and t.

    The function distinguishes between nodes that represent constant nodes in the tree
    (e.g. a pybamm.Matrix), and those that are variable (e.g. subtrees that contain
    pybamm.StateVector). The former are put in `constant_symbols`, the latter in
    `variable_symbols`

    Note that it is important that the arguments `constant_symbols` and
    `variable_symbols` be an *ordered* dict, since the final ordering of the code lines
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

    output_jax: bool
        If True, only numpy and jax operations will be used in the generated code,
        raises NotImplNotImplementedError if any SparseStack or Mat-Mat multiply
        operations are used

    """
    # constant symbols that are not numbers are stored in a list of constants, which are
    # passed into the generated function constant symbols that are numbers are written
    # directly into the code
    if symbol.is_constant():
        value = symbol.evaluate()
        if not isinstance(value, numbers.Number):
            if output_jax and scipy.sparse.issparse(value):
                # convert any remaining sparse matrices to our custom coo matrix
                constant_symbols[symbol.id] = create_jax_coo_matrix(value)
            else:
                constant_symbols[symbol.id] = value
        return

    # process children recursively
    for child in symbol.children:
        find_symbols(child, constant_symbols, variable_symbols, output_jax)

    # calculate the variable names that will hold the result of calculating the
    # children variables
    children_vars = []
    for child in symbol.children:
        if child.is_constant():
            child_eval = child.evaluate()
            if isinstance(child_eval, numbers.Number):
                children_vars.append(str(child_eval))
            else:
                children_vars.append(id_to_python_variable(child.id, True))
        else:
            children_vars.append(id_to_python_variable(child.id, False))

    if isinstance(symbol, pybamm.BinaryOperator):
        # Multiplication and Division need special handling for scipy sparse matrices
        # TODO: we can pass through a dummy y and t to get the type and then hardcode
        # the right line, avoiding these checks
        if isinstance(symbol, pybamm.Multiplication):
            dummy_eval_left = symbol.children[0].evaluate_for_shape()
            dummy_eval_right = symbol.children[1].evaluate_for_shape()
            if scipy.sparse.issparse(dummy_eval_left):
                if output_jax and is_scalar(dummy_eval_right):
                    symbol_str = (
                        f"{children_vars[0]}.scalar_multiply({children_vars[1]})"
                    )
                else:
                    symbol_str = f"{children_vars[0]}.multiply({children_vars[1]})"
            elif scipy.sparse.issparse(dummy_eval_right):
                symbol_str = f"{children_vars[1]}.multiply({children_vars[0]})"
            else:
                symbol_str = f"{children_vars[0]} * {children_vars[1]}"
        elif isinstance(symbol, pybamm.Division):
            dummy_eval_left = symbol.children[0].evaluate_for_shape()
            dummy_eval_right = symbol.children[1].evaluate_for_shape()
            if scipy.sparse.issparse(dummy_eval_left):
                if output_jax and is_scalar(dummy_eval_right):
                    symbol_str = (
                        f"{children_vars[0]}.scalar_multiply(1/{children_vars[1]})"
                    )
                else:
                    symbol_str = f"{children_vars[0]}.multiply(1/{children_vars[1]})"
            else:
                symbol_str = f"{children_vars[0]} / {children_vars[1]}"

        elif isinstance(symbol, pybamm.Inner):
            dummy_eval_left = symbol.children[0].evaluate_for_shape()
            dummy_eval_right = symbol.children[1].evaluate_for_shape()
            if scipy.sparse.issparse(dummy_eval_left):
                if output_jax and is_scalar(dummy_eval_right):
                    symbol_str = (
                        f"{children_vars[0]}.scalar_multiply({children_vars[1]})"
                    )
                else:
                    symbol_str = f"{children_vars[0]}.multiply({children_vars[1]})"
            elif scipy.sparse.issparse(dummy_eval_right):
                if output_jax and is_scalar(dummy_eval_left):
                    symbol_str = (
                        f"{children_vars[1]}.scalar_multiply({children_vars[0]})"
                    )
                else:
                    symbol_str = f"{children_vars[1]}.multiply({children_vars[0]})"
            else:
                symbol_str = f"{children_vars[0]} * {children_vars[1]}"

        elif isinstance(symbol, pybamm.Minimum):
            symbol_str = f"np.minimum({children_vars[0]},{children_vars[1]})"
        elif isinstance(symbol, pybamm.Maximum):
            symbol_str = f"np.maximum({children_vars[0]},{children_vars[1]})"

        elif isinstance(symbol, pybamm.MatrixMultiplication):
            dummy_eval_left = symbol.children[0].evaluate_for_shape()
            dummy_eval_right = symbol.children[1].evaluate_for_shape()
            if output_jax and (
                scipy.sparse.issparse(dummy_eval_left)
                and scipy.sparse.issparse(dummy_eval_right)
            ):
                raise NotImplementedError(
                    "sparse mat-mat multiplication not supported for output_jax == True"
                )
            else:
                symbol_str = (
                    children_vars[0] + " " + symbol.name + " " + children_vars[1]
                )
        else:
            symbol_str = children_vars[0] + " " + symbol.name + " " + children_vars[1]

    elif isinstance(symbol, pybamm.UnaryOperator):
        # Index has a different syntax than other univariate operations
        if isinstance(symbol, pybamm.Index):
            symbol_str = f"{children_vars[0]}[{symbol.slice.start}:{symbol.slice.stop}]"
        else:
            symbol_str = symbol.name + "(" + children_vars[0] + ")"

    elif isinstance(symbol, pybamm.Function):
        children_str = ""
        for child_var in children_vars:
            if children_str == "":
                children_str = child_var
            else:
                children_str += ", " + child_var
        if isinstance(symbol.function, np.ufunc):
            # write any numpy functions directly
            symbol_str = f"np.{symbol.function.__name__}({children_str})"
        else:
            # unknown function, store it as a constant and call this in the
            # generated code
            constant_symbols[symbol.id] = symbol.function
            funct_var = id_to_python_variable(symbol.id, True)
            symbol_str = f"{funct_var}({children_str})"

    elif isinstance(symbol, pybamm.Concatenation):
        # no need to concatenate if there is only a single child
        if isinstance(symbol, pybamm.NumpyConcatenation):
            if len(children_vars) == 1:
                symbol_str = children_vars[0]
            else:
                symbol_str = "np.concatenate(({}))".format(",".join(children_vars))

        elif isinstance(symbol, pybamm.SparseStack):
            if len(children_vars) == 1:
                symbol_str = children_vars[0]
            else:
                if output_jax:
                    raise NotImplementedError
                else:
                    symbol_str = "scipy.sparse.vstack(({}))".format(
                        ",".join(children_vars)
                    )

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
                            f"{child_var}[{child_slice[i].start}:{child_slice[i].stop}]"
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
        consecutive = np.all(indices[1:] - indices[:-1] == 1)
        if len(indices) == 1 or consecutive:
            symbol_str = f"y[{indices[0]}:{indices[-1] + 1}]"
        else:
            indices_array = pybamm.Array(indices)
            constant_symbols[indices_array.id] = indices
            index_name = id_to_python_variable(indices_array.id, True)
            symbol_str = f"y[{index_name}]"

    elif isinstance(symbol, pybamm.Time):
        symbol_str = "t"

    elif isinstance(symbol, pybamm.InputParameter):
        symbol_str = f'inputs["{symbol.name}"]'

    else:
        raise NotImplementedError(
            f"Conversion to python not implemented for a symbol of type '{type(symbol)}'"
        )

    variable_symbols[symbol.id] = symbol_str


def to_python(
    symbol: pybamm.Symbol, debug=False, output_jax=False
) -> tuple[OrderedDict, str]:
    """
    This function converts an expression tree into a dict of constant input values, and
    valid python code that acts like the tree's :func:`pybamm.Symbol.evaluate` function

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The symbol to convert to python code

    debug : bool
        If set to True, the function also emits debug code

    Returns
    -------
    collections.OrderedDict:
        dict mapping node id to a constant value. Represents all the constant nodes in
        the expression tree
    str:
        valid python code that will evaluate all the variable nodes in the tree.
    output_jax: bool
        If True, only numpy and jax operations will be used in the generated code.
        Raises NotImplNotImplementedError if any SparseStack or Mat-Mat multiply
        operations are used

    """
    constant_values: OrderedDict = OrderedDict()
    variable_symbols: OrderedDict = OrderedDict()
    find_symbols(symbol, constant_values, variable_symbols, output_jax)

    line_format = "{} = {}"

    if debug:  # pragma: no cover
        variable_lines = [
            f"print('{line_format.format(id_to_python_variable(symbol_id, False), symbol_line)}'); "
            + line_format.format(id_to_python_variable(symbol_id, False), symbol_line)
            + "; print(type({0}),np.shape({0}))".format(
                id_to_python_variable(symbol_id, False)
            )
            for symbol_id, symbol_line in variable_symbols.items()
        ]
    else:
        variable_lines = [
            line_format.format(id_to_python_variable(symbol_id, False), symbol_line)
            for symbol_id, symbol_line in variable_symbols.items()
        ]

    return constant_values, "\n".join(variable_lines)


class EvaluatorPython:
    """
    Converts a pybamm expression tree into pure python code that will calculate the
    result of calling `evaluate(t, y)` on the given expression tree.

    Parameters
    ----------

    symbol : :class:`pybamm.Symbol`
        The symbol to convert to python code


    """

    def __init__(self, symbol: pybamm.Symbol):
        constants, python_str = pybamm.to_python(symbol, debug=False)

        # extract constants in generated function
        for i, symbol_id in enumerate(constants.keys()):
            const_name = id_to_python_variable(symbol_id, True)
            python_str = f"{const_name} = constants[{i}]\n" + python_str

        # constants passed in as an ordered dict, convert to list
        self._constants = list(constants.values())

        # indent code
        python_str = "   " + python_str
        python_str = python_str.replace("\n", "\n   ")

        # add function def to first line
        python_str = (
            "def evaluate(constants, t=None, y=None, inputs=None):\n" + python_str
        )

        # calculate the final variable that will output the result of calling `evaluate`
        # on `symbol`
        result_var = id_to_python_variable(symbol.id, symbol.is_constant())
        if symbol.is_constant():
            result_value = symbol.evaluate()

        # add return line
        if symbol.is_constant() and isinstance(result_value, numbers.Number):
            python_str = python_str + "\n   return " + str(result_value)
        else:
            python_str = python_str + "\n   return " + result_var

        # store a copy of examine_jaxpr
        python_str = python_str + "\nself._evaluate = evaluate"

        self._python_str = python_str
        self._result_var = result_var
        self._symbol = symbol

        # compile and run the generated python code,
        compiled_function = compile(python_str, result_var, "exec")
        exec(compiled_function)

    def __call__(self, t=None, y=None, inputs=None):
        """
        evaluate function
        """
        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        result = self._evaluate(self._constants, t, y, inputs)

        return result

    def __getstate__(self):
        # Control the state of instances of EvaluatorPython
        # before pickling. Method "_evaluate" cannot be pickled.
        # See https://github.com/pybamm-team/PyBaMM/issues/1283
        state = self.__dict__.copy()
        del state["_evaluate"]
        return state

    def __setstate__(self, state):
        # Restore pickled attributes and
        # compile code from "python_str"
        # Execution of bytecode (re)adds attribute
        # "_method"
        self.__dict__.update(state)
        compiled_function = compile(self._python_str, self._result_var, "exec")
        exec(compiled_function)


class EvaluatorJax:
    """
    Converts a pybamm expression tree into pure python code that will calculate the
    result of calling `evaluate(t, y)` on the given expression tree. The resultant code
    is compiled with JAX

    Limitations: JAX currently does not work on expressions involving sparse matrices,
    so any sparse matrices and operations involved sparse matrices are converted to
    their dense equivilents before compilation

    Parameters
    ----------

    symbol : :class:`pybamm.Symbol`
        The symbol to convert to python code


    """

    def __init__(self, symbol: pybamm.Symbol):
        if not pybamm.has_jax():  # pragma: no cover
            raise ModuleNotFoundError(
                "Jax or jaxlib is not installed, please see https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver"
            )

        constants, python_str = pybamm.to_python(symbol, debug=False, output_jax=True)

        # replace numpy function calls to jax numpy calls
        python_str = python_str.replace("np.", "jax.numpy.")

        # convert all numpy constants to device vectors
        for symbol_id in constants:
            if isinstance(constants[symbol_id], np.ndarray):
                constants[symbol_id] = jax.device_put(constants[symbol_id])

        # get a list of constant arguments to input to the function
        self._arg_list = [
            id_to_python_variable(symbol_id, True) for symbol_id in constants.keys()
        ]

        # get a list of hashable arguments to make static
        # a jax device array is not hashable
        static_argnums = (
            i
            for i, c in enumerate(constants.values())
            if not (isinstance(c, jax.Array))
        )

        # store constants
        self._constants = tuple(constants.values())

        # indent code
        python_str = "   " + python_str
        python_str = python_str.replace("\n", "\n   ")

        # add function def to first line
        args = "t=None, y=None, inputs=None"
        if self._arg_list:
            args = ",".join(self._arg_list) + ", " + args
        python_str = f"def evaluate_jax({args}):\n" + python_str

        # calculate the final variable that will output the result of calling `evaluate`
        # on `symbol`
        result_var = id_to_python_variable(symbol.id, symbol.is_constant())
        if symbol.is_constant():
            result_value = symbol.evaluate()

        # add return line
        if symbol.is_constant() and isinstance(result_value, numbers.Number):
            python_str = python_str + "\n   return " + str(result_value)
        else:
            python_str = python_str + "\n   return " + result_var

        # store a copy of examine_jaxpr
        python_str = python_str + "\nself._evaluate_jax = evaluate_jax"

        # store the final generated code
        self._python_str = python_str

        # compile and run the generated python code,
        compiled_function = compile(python_str, result_var, "exec")
        exec(compiled_function)

        self._static_argnums = tuple(static_argnums)
        self._jit_evaluate = jax.jit(
            self._evaluate_jax,  # type:ignore[attr-defined]
            static_argnums=self._static_argnums,
        )

    def _demote_constants(self):
        """Demote 64-bit constants (f64, i64) to 32-bit (f32, i32)"""
        if not pybamm.demote_expressions_to_32bit:
            return  # pragma: no cover
        self._constants = EvaluatorJax._demote_64_to_32(self._constants)

    @classmethod
    def _demote_64_to_32(cls, c):
        """Demote 64-bit operations (f64, i64) to 32-bit (f32, i32)"""

        if not pybamm.demote_expressions_to_32bit:
            return c
        if isinstance(c, float):
            c = jax.numpy.float32(c)
        if isinstance(c, int):
            c = jax.numpy.int32(c)
        if isinstance(c, np.int64):
            c = c.astype(jax.numpy.int32)
        if isinstance(c, np.ndarray):
            if c.dtype == np.float64:
                c = c.astype(jax.numpy.float32)
            if c.dtype == np.int64:
                c = c.astype(jax.numpy.int32)
        if isinstance(c, jax.numpy.ndarray):
            if c.dtype == jax.numpy.float64:
                c = c.astype(jax.numpy.float32)
            if c.dtype == jax.numpy.int64:
                c = c.astype(jax.numpy.int32)
        if isinstance(
            c, pybamm.expression_tree.operations.evaluate_python.JaxCooMatrix
        ):
            if c.data.dtype == np.float64:
                c.data = c.data.astype(jax.numpy.float32)
            if c.row.dtype == np.int64:
                c.row = c.row.astype(jax.numpy.int32)
            if c.col.dtype == np.int64:
                c.col = c.col.astype(jax.numpy.int32)
        if isinstance(c, dict):
            c = {key: EvaluatorJax._demote_64_to_32(value) for key, value in c.items()}
        if isinstance(c, tuple):
            c = tuple(EvaluatorJax._demote_64_to_32(value) for value in c)
        if isinstance(c, list):
            c = [EvaluatorJax._demote_64_to_32(value) for value in c]
        return c

    @property
    def _constants(self):
        return tuple(map(EvaluatorJax._demote_64_to_32, self.__constants))

    @_constants.setter
    def _constants(self, value):
        self.__constants = value

    def get_jacobian(self):
        n = len(self._arg_list)

        # forward mode autodiff  wrt y, which is argument 1 after arg_list
        jacobian_evaluate = jax.jacfwd(self._evaluate_jax, argnums=1 + n)

        self._jac_evaluate = jax.jit(
            jacobian_evaluate, static_argnums=self._static_argnums
        )

        return EvaluatorJaxJacobian(self._jac_evaluate, self._constants)

    def get_jacobian_action(self):
        return self.jvp

    def get_sensitivities(self):
        n = len(self._arg_list)

        # forward mode autodiff wrt inputs, which is argument 2 after arg_list
        jacobian_evaluate = jax.jacfwd(self._evaluate_jax, argnums=2 + n)

        self._sens_evaluate = jax.jit(
            jacobian_evaluate, static_argnums=self._static_argnums
        )

        return EvaluatorJaxSensitivities(self._sens_evaluate, self._constants)

    def debug(self, t=None, y=None, inputs=None):
        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        # execute code
        jaxpr = jax.make_jaxpr(self._evaluate_jax)(*self._constants, t, y, inputs).jaxpr
        print("invars:", jaxpr.invars)
        print("outvars:", jaxpr.outvars)
        print("constvars:", jaxpr.constvars)
        for eqn in jaxpr.eqns:
            print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
        print()
        print("jaxpr:", jaxpr)

    def __call__(self, t=None, y=None, inputs=None):
        """
        evaluate function
        """
        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        result = self._jit_evaluate(*self._constants, t, y, inputs)

        return result

    def jvp(self, t=None, y=None, v=None, inputs=None):
        """
        evaluate jacobian vector product of function
        """

        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)
        if v is not None and v.ndim == 1:
            v = v.reshape(-1, 1)

        def bind_t_and_inputs(the_y):
            return self._jit_evaluate(*self._constants, t, the_y, inputs)

        return jax.jvp(bind_t_and_inputs, (y,), (v,))[1]


class EvaluatorJaxJacobian:
    def __init__(self, jac_evaluate, constants):
        self._jac_evaluate = jac_evaluate
        self._constants = constants

    def __call__(self, t=None, y=None, inputs=None):
        """
        evaluate function
        """
        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        # execute code
        result = self._jac_evaluate(*self._constants, t, y, inputs)
        result = result.reshape(result.shape[0], -1)

        return result


class EvaluatorJaxSensitivities:
    def __init__(self, jac_evaluate, constants):
        self._jac_evaluate = jac_evaluate
        self._constants = constants

    def __call__(self, t=None, y=None, inputs=None):
        """
        evaluate function
        """
        # generated code assumes y is a column vector
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        # execute code
        result = self._jac_evaluate(*self._constants, t, y, inputs)
        result = {
            key: value.reshape(value.shape[0], -1) for key, value in result.items()
        }

        return result
