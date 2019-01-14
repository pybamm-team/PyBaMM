#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import copy


class BaseDiscretisation(object):
    """The base discretisation class, with methods to process a model and replace
    Spatial Operators with Matrices and Variables with StateVectors

    Parameters
    ----------
    mesh : :class:`pybamm.BaseMesh` (or subclass)
        The underlying mesh for discretisation

    """

    def __init__(self, mesh):
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def process_model(self, model):
        """Discretise a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        """
        # Set the y split for variables
        y_slices = self.get_variable_slices(model.rhs.keys())

        # Discretise and concatenate initial conditions, passing domain from variable
        model.initial_conditions = self.process_initial_conditions(
            model.initial_conditions
        )

        # Discretise right-hand sides, passing domain from variable
        model.rhs = self.process_rhs(model.rhs, model.boundary_conditions, y_slices)

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
                end += self.mesh[dom].npts
            y_slices[variable.id] = slice(start, end)
            start = end

        return y_slices

    def process_initial_conditions(self, initial_conditions):
        """Discretise initial conditions.

        Parameters
        ----------
        initial_conditions : dict
            Initial conditions ({variable: equation} dict) to dicretise

        Returns
        -------
        :class:`numpy.array`
            Vector of initial conditions

        """
        for variable, equation in initial_conditions.items():
            discretised_ic = self.process_symbol(equation, variable.domain)
            # Turn any scalars into vectors
            if isinstance(discretised_ic, pybamm.Scalar):
                discretised_ic = self.scalar_to_vector(discretised_ic, variable.domain)
            initial_conditions[variable] = discretised_ic

        # Concatenate and evaluate initial conditions
        return self.concatenate(*initial_conditions.values()).evaluate(0, None)

    def process_rhs(self, rhs, boundary_conditions, y_slices):
        """Discretise initial conditions.

        Parameters
        ----------
        rhs : dict
            Equations ({variable: equation} dict) to dicretise
        boundary_conditions : dict
            Boundary conditions ({symbol.id: {"left": left bc, "right": right bc}} dict)
            associated with rhs to dicretise
        y_slices : dict of {variable id: slice}
            The slices to assign to StateVectors when discretising a variable

        Returns
        -------
        concatenated_rhs : :class:`pybamm.Concatenation`
            Concatenation of the discretised right-hand side equations

        """
        boundary_conditions = {
            key.id: value for key, value in boundary_conditions.items()
        }
        for variable, equation in rhs.items():
            rhs[variable] = self.process_symbol(
                equation, variable.domain, y_slices, boundary_conditions
            )

        # Concatenate right-hand sides
        concatenated_rhs = self.concatenate(*rhs.values())

        return concatenated_rhs

    def process_symbol(self, symbol, domain, y_slices=None, boundary_conditions={}):
        """Discretise operators in model equations.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol to discretise
        y_slices : dict of {Variable: slice}
            The slices to assign to StateVectors when discretising a variable

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
            left, right = symbol.children
            new_left = self.process_symbol(left, domain, y_slices, boundary_conditions)
            new_right = self.process_symbol(
                right, domain, y_slices, boundary_conditions
            )
            return symbol.__class__(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(
                symbol.children[0], domain, y_slices, boundary_conditions
            )
            return symbol.__class__(new_child)

        elif isinstance(symbol, pybamm.Variable):
            assert isinstance(y_slices, dict), ValueError(
                """y_slices should be dict, not {}""".format(type(y_slices))
            )
            return pybamm.StateVector(y_slices[symbol.id])

        else:
            # hack to copy the symbol but without a parent
            # (building tree from bottom up)
            # simply setting new_symbol.parent = None, after copying, raises a TreeError
            parent = symbol.parent
            symbol.parent = None
            new_symbol = copy.copy(symbol)
            symbol.parent = parent
            return new_symbol

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """How to discretise gradient operators.

        Parameters
        ----------
        symbol : :class:`Symbol` (or subclass)
            The symbol (typically a variable) of which to take the gradient
        domain : list
            The domain(s) in which to take the gradient
        y_slices : slice
            The slice to assign to StateVector when discretising a variable
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        """
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        """How to discretise divergence operators.

        Parameters
        ----------
        symbol : :class:`Symbol` (or subclass)
            The symbol (typically a variable) of which to take the divergence
        domain : list
            The domain(s) in which to take the divergence
        y_slices : slice
            The slice to assign to StateVector when discretising a variable
        boundary_conditions : dict
            The boundary conditions of the model

        """
        raise NotImplementedError

    def scalar_to_vector(self, scalar, domain):
        """
        Convert a Scalar to a uniform Vector of size given by mesh,
        with same value as Scalar.
        """
        mesh_points = np.array([])
        for dom in domain:
            mesh_points = np.concatenate([mesh_points, self.mesh[dom].nodes])
        return pybamm.Vector(scalar.value * np.ones_like(mesh_points))

    def concatenate(self, *symbols):
        return pybamm.NumpyConcatenation(*symbols)
