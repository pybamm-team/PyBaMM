#
# Convert a PyBaMM expression tree to a CasADi expression tree
#
import pybamm
import casadi
import numpy as np
from scipy import special


class CasadiConverter(object):
    def __init__(self, casadi_symbols=None):
        self._casadi_symbols = casadi_symbols or {}

        pybamm.citations.register("Andersson2019")

    def convert(self, symbol, t, y, y_dot, inputs):
        """
        This function recurses down the tree, converting the PyBaMM expression tree to
        a CasADi expression tree

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to convert
        t : :class:`casadi.MX`
            A casadi symbol representing time
        y : :class:`casadi.MX`
            A casadi symbol representing state vectors
        y_dot : :class:`casadi.MX`
            A casadi symbol representing time derivatives of state vectors
        inputs : dict
            A dictionary of casadi symbols representing parameters

        Returns
        -------
        :class:`casadi.MX`
            The converted symbol
        """
        try:
            return self._casadi_symbols[symbol.id]
        except KeyError:
            # Change inputs to empty dictionary if it's None
            inputs = inputs or {}
            casadi_symbol = self._convert(symbol, t, y, y_dot, inputs)
            self._casadi_symbols[symbol.id] = casadi_symbol

            return casadi_symbol

    def _convert(self, symbol, t, y, y_dot, inputs):
        """ See :meth:`CasadiConverter.convert()`. """
        if isinstance(
            symbol,
            (
                pybamm.Scalar,
                pybamm.Array,
                pybamm.Time,
                pybamm.InputParameter,
                pybamm.ExternalVariable,
            ),
        ):
            return casadi.MX(symbol.evaluate(t, y, y_dot, inputs))

        elif isinstance(symbol, pybamm.StateVector):
            if y is None:
                raise ValueError("Must provide a 'y' for converting state vectors")
            return casadi.vertcat(*[y[y_slice] for y_slice in symbol.y_slices])

        elif isinstance(symbol, pybamm.StateVectorDot):
            if y_dot is None:
                raise ValueError("Must provide a 'y_dot' for converting state vectors")
            return casadi.vertcat(*[y_dot[y_slice] for y_slice in symbol.y_slices])

        elif isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            converted_left = self.convert(left, t, y, y_dot, inputs)
            converted_right = self.convert(right, t, y, y_dot, inputs)

            if isinstance(symbol, pybamm.Modulo):
                return casadi.fmod(converted_left, converted_right)
            if isinstance(symbol, pybamm.Minimum):
                return casadi.fmin(converted_left, converted_right)
            if isinstance(symbol, pybamm.Maximum):
                return casadi.fmax(converted_left, converted_right)

            # _binary_evaluate defined in derived classes for specific rules
            return symbol._binary_evaluate(converted_left, converted_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            converted_child = self.convert(symbol.child, t, y, y_dot, inputs)
            if isinstance(symbol, pybamm.AbsoluteValue):
                return casadi.fabs(converted_child)
            if isinstance(symbol, pybamm.Floor):
                return casadi.floor(converted_child)
            if isinstance(symbol, pybamm.Ceiling):
                return casadi.ceil(converted_child)
            return symbol._unary_evaluate(converted_child)

        elif isinstance(symbol, pybamm.Function):
            converted_children = [
                self.convert(child, t, y, y_dot, inputs) for child in symbol.children
            ]
            # Special functions
            if symbol.function == np.min:
                return casadi.mmin(*converted_children)
            elif symbol.function == np.max:
                return casadi.mmax(*converted_children)
            elif symbol.function == np.abs:
                return casadi.fabs(*converted_children)
            elif symbol.function == np.sqrt:
                return casadi.sqrt(*converted_children)
            elif symbol.function == np.sin:
                return casadi.sin(*converted_children)
            elif symbol.function == np.arcsinh:
                return casadi.arcsinh(*converted_children)
            elif symbol.function == np.arccosh:
                return casadi.arccosh(*converted_children)
            elif symbol.function == np.tanh:
                return casadi.tanh(*converted_children)
            elif symbol.function == np.cosh:
                return casadi.cosh(*converted_children)
            elif symbol.function == np.sinh:
                return casadi.sinh(*converted_children)
            elif symbol.function == np.cos:
                return casadi.cos(*converted_children)
            elif symbol.function == np.exp:
                return casadi.exp(*converted_children)
            elif symbol.function == np.log:
                return casadi.log(*converted_children)
            elif symbol.function == np.sign:
                return casadi.sign(*converted_children)
            elif symbol.function == special.erf:
                return casadi.erf(*converted_children)
            elif isinstance(symbol, pybamm.Interpolant):
                return casadi.interpolant(
                    "LUT", "bspline", symbol.x, symbol.y.flatten()
                )(*converted_children)
            elif symbol.function.__name__.startswith("elementwise_grad_of_"):
                differentiating_child_idx = int(symbol.function.__name__[-1])
                # Create dummy symbolic variables in order to differentiate using CasADi
                dummy_vars = [
                    casadi.MX.sym("y_" + str(i)) for i in range(len(converted_children))
                ]
                func_diff = casadi.gradient(
                    symbol.differentiated_function(*dummy_vars),
                    dummy_vars[differentiating_child_idx],
                )
                # Create function and evaluate it using the children
                casadi_func_diff = casadi.Function("func_diff", dummy_vars, [func_diff])
                return casadi_func_diff(*converted_children)
            # Other functions
            else:
                return symbol._function_evaluate(converted_children)
        elif isinstance(symbol, pybamm.Concatenation):
            converted_children = [
                self.convert(child, t, y, y_dot, inputs) for child in symbol.children
            ]
            if isinstance(symbol, (pybamm.NumpyConcatenation, pybamm.SparseStack)):
                return casadi.vertcat(*converted_children)
            # DomainConcatenation specifies a particular ordering for the concatenation,
            # which we must follow
            elif isinstance(symbol, pybamm.DomainConcatenation):
                slice_starts = []
                all_child_vectors = []
                for i in range(symbol.secondary_dimensions_npts):
                    child_vectors = []
                    for child_var, slices in zip(
                        converted_children, symbol._children_slices
                    ):
                        for child_dom, child_slice in slices.items():
                            slice_starts.append(symbol._slices[child_dom][i].start)
                            child_vectors.append(
                                child_var[child_slice[i].start : child_slice[i].stop]
                            )
                    all_child_vectors.extend(
                        [v for _, v in sorted(zip(slice_starts, child_vectors))]
                    )
                return casadi.vertcat(*all_child_vectors)

        else:
            raise TypeError(
                """
                Cannot convert symbol of type '{}' to CasADi. Symbols must all be
                'linear algebra' at this stage.
                """.format(
                    type(symbol)
                )
            )
