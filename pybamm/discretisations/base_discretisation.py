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
        model.rhs = self.process_rhs(model.rhs, model.boundary_conditions, y_slices)

        return y0

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
            Boundary conditions ({symbol.id: (left bc, right bc)} dict) associated with
            rhs to dicretise
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
            new_left = self.process_symbol(
                symbol.children[0], domain, y_slices, boundary_conditions
            )
            new_right = self.process_symbol(
                symbol.children[1], domain, y_slices, boundary_conditions
            )
            return symbol.__class__(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(
                symbol.children[0], domain, y_slices, boundary_conditions
            )
            return symbol.__class__(new_child)

        elif isinstance(symbol, pybamm.Variable):
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
            The boundary conditions of the model ({symbol.id: (left bc, right bc)})

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


class MatrixVectorDiscretisation(BaseDiscretisation):
    """
    A base class for discretisations where the gradient and divergence operators are
    always matrix-vector multiplications.

    Parameters
    ----------
     mesh : :class:`pybamm.BaseMesh` (or subclass)
            The underlying mesh for discretisation

    **Extends:** :class:`pybamm.BaseDiscretisation`
    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
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
        """The gradient matrix

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix
        """
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.BaseDiscretisation.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        discretised_symbol = self.process_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol.id in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol.id]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        divergence_matrix = self.divergence_matrix(domain)
        return divergence_matrix * discretised_symbol

    def divergence_matrix(self, domain):
        """The divergence matrix

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix
        """
        raise NotImplementedError
