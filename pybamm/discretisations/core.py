#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import copy
import numpy as np


class BaseDiscretisation(object):
    def __init__(self, mesh):
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """How to discretise gradient operators"""
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
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

        # Discretise right-hand sides, passing domain from variable
        for variable, equation in model.rhs.items():
            model.rhs[variable] = self.discretise_symbol(
                equation, variable.domain, y_slices, model.boundary_conditions
            )
            # TODO: deal with boundary conditions

        # Discretise and concatenate initial conditions, passing domain from variable
        for variable, equation in model.initial_conditions.items():
            model.initial_conditions[variable] = self.discretise_symbol(
                equation, variable.domain, y_slices, model.boundary_conditions
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

    def discretise_symbol(self, symbol, domain, y_slices, boundary_conditions):
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
            return self.gradient(symbol.child, domain, y_slices, boundary_conditions)

        if isinstance(symbol, pybamm.Divergence):
            return self.divergence(symbol.child, domain, y_slices, boundary_conditions)

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_binary_operator = copy.copy(symbol)
            new_binary_operator.left = self.discretise_symbol(
                symbol.left, domain, y_slices, boundary_conditions
            )
            new_binary_operator.right = self.discretise_symbol(
                symbol.right, domain, y_slices, boundary_conditions
            )
            return new_binary_operator

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_unary_operator = copy.copy(symbol)
            new_unary_operator.child = self.discretise_symbol(
                symbol.child, domain, y_slices, boundary_conditions
            )
            return new_unary_operator

        elif isinstance(symbol, pybamm.Variable):
            return pybamm.Vector(y_slices[symbol])

        elif isinstance(symbol, pybamm.Value):
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

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Not necessarily a Variable (e.g. N = -D(c) * grad(c) is a Multiplication)
        """
        gradient_matrix = self.gradient_matrix(domain)
        discretised_symbol = self.discretise_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        return gradient_matrix * discretised_symbol

    def gradient_matrix(self, domain):
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        gradient_matrix = self.gradient_matrix(domain)
        discretised_symbol = self.discretise_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        return gradient_matrix * discretised_symbol

    def divergence_matrix(self, domain):
        raise NotImplementedError

    def concatenate(self, *symbols):
        # overwrite evaluation of the
        return pybamm.NumpyConcatenation(*symbols)
