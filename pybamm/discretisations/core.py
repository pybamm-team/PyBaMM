#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

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

    def process_model(self, model):
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
        y_slices = self.get_variable_slices(model.rhs.keys())

        # Discretise and concatenate initial conditions, passing domain from variable
        y0 = self.process_initial_conditions(model.initial_conditions)

        # Discretise right-hand sides, passing domain from variable
        dydt = self.process_rhs(model.rhs, model.boundary_conditions, y_slices)

        return y0, dydt

    def get_variable_slices(self, variables):
        """Set the slicing for variables.

        Parameters
        ----------
        variables : dict_keys object containing Variable instances
            Variables for which to set the slices

        Returns
        -------
        y_slices : dict of {variable id: slice}
            The slices to take when solving (assigning chunks of y to each vector)
        """
        y_slices = {variable.id: None for variable in variables}
        start = 0
        end = 0
        for variable in variables:
            # Add up the size of all the domains in variable.domain
            for dom in variable.domain:
                end += getattr(self.mesh, dom).npts
            y_slices[variable.id] = slice(start, end)
            start = end

        return y_slices

    def process_initial_conditions(self, initial_conditions):
        for variable, equation in initial_conditions.items():
            discretised_ic = self.process_symbol(equation, variable.domain)
            # Turn any scalars into vectors
            if isinstance(discretised_ic, pybamm.Scalar):
                discretised_ic = self.scalar_to_vector(discretised_ic, variable.domain)
            initial_conditions[variable] = discretised_ic

        # Concatenate and evaluate initial conditions
        return pybamm.NumpyConcatenation(*initial_conditions.values()).evaluate(0, None)

    def process_rhs(self, rhs, boundary_conditions, y_slices):
        boundary_conditions = {
            key.id: value for key, value in boundary_conditions.items()
        }
        for variable, equation in rhs.items():
            rhs[variable] = self.process_symbol(
                equation, variable.domain, y_slices, boundary_conditions
            )

        # Concatenate and evaluate right-hand sides
        self._concatenated_rhs = pybamm.NumpyConcatenation(*rhs.values())

        def dydt(t, y):
            return self._concatenated_rhs.evaluate(t, y)

        return dydt

    def process_symbol(self, symbol, domain, y_slices=None, boundary_conditions=None):
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
            return self.gradient(
                symbol.children[0], domain, y_slices, boundary_conditions
            )

        if isinstance(symbol, pybamm.Divergence):
            return self.divergence(
                symbol.children[0], domain, y_slices, boundary_conditions
            )

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_left = self.process_symbol(
                symbol.children[0], domain, y_slices, boundary_conditions
            )
            new_right = self.process_symbol(
                symbol.children[1], domain, y_slices, boundary_conditions
            )
            return symbol.__class__(symbol.name, new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(
                symbol.children[0], domain, y_slices, boundary_conditions
            )
            return symbol.__class__(symbol.name, new_child)

        elif isinstance(symbol, pybamm.Variable):
            return pybamm.VariableVector(y_slices[symbol.id])

        elif isinstance(symbol, pybamm.Scalar):
            return pybamm.Scalar(symbol.value)

        else:
            raise TypeError("""Cannot discretise {!r}""".format(symbol))

    def scalar_to_vector(self, scalar, domain):
        """
        Convert a Scalar to a uniform Vector of size given by mesh,
        with same value as Scalar.
        """
        mesh_points = np.array([])
        for dom in domain:
            mesh_points = np.concatenate([mesh_points, getattr(self.mesh, dom).centres])
        return pybamm.Vector(scalar.value * np.ones_like(mesh_points))

    def concatenate(self, *symbols):
        return pybamm.NumpyConcatenation(*symbols)


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
        discretised_symbol = self.process_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol.id in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol.id]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        gradient_matrix = self.gradient_matrix(domain)
        return gradient_matrix * discretised_symbol

    def gradient_matrix(self, domain):
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        discretised_symbol = self.process_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol.id in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol.id]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        divergence_matrix = self.gradient_matrix(domain)
        return divergence_matrix * discretised_symbol

    def divergence_matrix(self, domain):
        raise NotImplementedError
