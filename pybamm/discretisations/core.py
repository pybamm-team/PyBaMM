#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import copy


class BaseDiscretisation(object):
    def __init__(self, mesh):
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def gradient(self):
        """How to discretise gradient operators"""
        raise NotImplementedError

    def divergence(self):
        """How to discretise divergence operators"""
        raise NotImplementedError

    def discretise_model(self, model):
        """Discretise a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : dict
            Model ({variable: equation} dict) to dicretise

        Returns
        -------
        y0 : :class:`numpy.array`
            Vector of initial conditions

        """
        # Set the y split for variables
        y_slices = self.get_variable_slices(model.keys())

        # Discretise right-hand sides
        for variable, equation in model.rhs.items():
            model.rhs[variable] = self.discretise_symbol(equation, y_slices)
            # TODO: deal with boundary conditions

        # Discretise and concatenate initial conditions
        for variable, equation in model.initial_conditions.items():
            model.initial_conditions[variable] = self.discretise_symbol(
                equation, y_slices
            )
            ics_equations = [ic for ic in model.initial_conditions.values()]
            y0 = np.concatenate([ic.value for ic in ics_equations])

        return y0

    def get_variable_slices(self, variables):
        """Set the slicing for variables.

        Parameters
        ----------
        variables : dict_keys object containing Variable instances
            Variables for which to set the slices

        Returns
        -------
        y_slices : dict of {Variable: slice}
            The slices to take when solving (assigning chunks of y to each vector)
        """
        y_slices = {variable: None for variable in variables}
        start = 0
        for variable in variables:
            end = start + getattr(self.mesh, variable.domain).npts
            y_slices[variable] = slice(start, end)
            start = end

        return y_slices

    def discretise_symbol(self, symbol, y_slices):
        """Discretise operators in model equations.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol to discretise
        y_slices : dict of {Variable: slice}
            The slices to take when solving (assigning chunks of y to each vector)

        Returns
        -------
        :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Discretised symbol

        """
        if isinstance(symbol, pybamm.Gradient):
            variable = symbol.child
            return self.gradient(variable, y_slices)

        if isinstance(symbol, pybamm.Divergence):
            variable = symbol.child
            return self.divergence(variable, y_slices)

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_symbol = copy.copy(symbol)
            new_symbol.left = self.discretise_symbol(symbol.left)
            new_symbol.right = self.discretise_symbol(symbol.right)
            return new_symbol

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_symbol = copy.copy(symbol)
            new_symbol.child = self.discretise_symbol(symbol.child)
            return new_symbol

        elif isinstance(symbol, pybamm.Variable) or isinstance(symbol, pybamm.Value):
            return copy.copy(symbol)

        else:
            raise TypeError("""Cannot discretise {!r}""".format(symbol))


class MatrixVectorDiscretisation(BaseDiscretisation):
    """
    A base class for discretisations where the gradient and divergence operators are
    always matrix-vector multiplications.
    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient(self, variable, y_slices):
        variable = symbol.child
        gradient_matrix = self.gradient_matrix(variable.domain)
        variable_vector = pybamm.Vector(y_slice=y_slices[variable])
        return gradient_matrix * variable_vector

    def gradient_matrix(self, domain):
        raise NotImplementedError

    def divergence(self, variable, y_slices):
        variable = symbol.child
        gradient_matrix = self.gradient_matrix(variable.domain)
        variable_vector = pybamm.Vector(y_slice=y_slices[variable])
        return gradient_matrix * variable_vector

    def divergence_matrix(self, domain):
        raise NotImplementedError
