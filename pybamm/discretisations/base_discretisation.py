#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import numbers
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

        # Discretise initial conditions
        model.initial_conditions = self.process_initial_conditions(
            model.initial_conditions
        )
        # Concatenate initial conditions into a single vector
        model.concatenated_initial_conditions = self.concatenate(
            *model.initial_conditions.values()
        ).evaluate(0, None)

        # Discretise right-hand sides, passing domain from variable
        model.rhs = self.process_dict(model.rhs, y_slices, model.boundary_conditions)
        # Concatenate rhs into a single state vector
        model.concatenated_rhs = self.concatenate(*model.rhs.values())

        # Discretise variables (applying boundary conditions)
        # Note that we **do not** discretise the keys of model.rhs,
        # model.initial_conditions and model.boundary_conditions
        model.variables = self.process_dict(
            model.variables, y_slices, model.boundary_conditions
        )

        self.check_model(model)

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
        initial_conditions : dict
            Discretised initial conditions

        """
        for variable, equation in initial_conditions.items():
            discretised_ic = self.process_symbol(equation).evaluate()

            if isinstance(discretised_ic, numbers.Number):
                discretised_ic = discretised_ic * self.vector_of_ones(variable.domain)
            else:
                raise NotImplementedError(
                    "Currently only accepts scalar initial conditions"
                )

            initial_conditions[variable] = discretised_ic

        return initial_conditions

    def process_dict(self, var_eqn_dict, y_slices, boundary_conditions):
        """Discretise a dictionary of {variable: equation}
        (can be model.rhs or model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
        y_slices : dict of {variable id: slice}
            The slices to assign to StateVectors when discretising
        boundary_conditions : dict
            Boundary conditions ({symbol.id: {"left": left bc, "right": right bc}} dict)
            associated with var_eqn_dict to dicretise

        Returns
        -------
        var_eqn_dict : dict
            Discretised right-hand side equations

        """
        boundary_conditions = {
            key.id: value for key, value in boundary_conditions.items()
        }
        for variable, equation in var_eqn_dict.items():
            var_eqn_dict[variable] = self.process_symbol(
                equation, y_slices, boundary_conditions
            )

        return var_eqn_dict

    def process_symbol(self, symbol, y_slices=None, boundary_conditions={}):
        """Discretise operators in model equations.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol to discretise
        y_slices : dict of {variable: slice}
            The slices to assign to StateVectors when discretising a variable
            (default None).
        boundary_conditions : dict of {variable: boundary conditions}
            Boundary conditions of the model

        Returns
        -------
        :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Discretised symbol

        """
        if isinstance(symbol, pybamm.Gradient):
            return self.gradient(symbol.children[0], y_slices, boundary_conditions)

        if isinstance(symbol, pybamm.Divergence):
            return self.divergence(symbol.children[0], y_slices, boundary_conditions)

        elif isinstance(symbol, pybamm.BinaryOperator):
            return self.process_binary_operators(symbol, y_slices, boundary_conditions)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(
                symbol.children[0], y_slices, boundary_conditions
            )
            return symbol.__class__(new_child)

        elif isinstance(symbol, pybamm.Variable):
            assert isinstance(y_slices, dict), ValueError(
                """y_slices should be dict, not {}""".format(type(y_slices))
            )
            return pybamm.StateVector(y_slices[symbol.id])

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [
                self.process_symbol(child, y_slices, boundary_conditions)
                for child in symbol.children
            ]
            new_symbol = pybamm.DomainConcatenation(new_children, self.mesh)

            if new_symbol.is_constant():
                return pybamm.Vector(new_symbol.evaluate())

            return new_symbol

        elif isinstance(symbol, pybamm.Scalar):
            return pybamm.Scalar(symbol.value, domain=symbol.domain)

        elif isinstance(symbol, pybamm.Array):
            return symbol.__class__(symbol.entries, domain=symbol.domain)

        else:
            # hack to copy the symbol but without a parent
            # (building tree from bottom up)
            # simply setting new_symbol.parent = None, after copying, raises a TreeError
            parent = symbol.parent
            symbol.parent = None
            new_symbol = copy.copy(symbol)
            symbol.parent = parent
            return new_symbol

    def process_binary_operators(self, bin_op, y_slices, boundary_conditions):
        """Discretise binary operators in model equations.
        Performs appropriate averaging of diffusivities if one of the children is a
        gradient operator, so that discretised sizes match up.
        This is mainly an issue for the Finite Volume Discretisation:
        see :meth:`pybamm.FiniteVolumeDiscretisation.compute_diffusivity()`

        Parameters
        ----------
        bin_op : :class:`pybamm.BinaryOperator` (or subclass)
            Binary operator to discretise
        y_slices : dict of {variable: slice}
            The slices to assign to StateVectors when discretising a variable
            (default None).
        boundary_conditions : dict of {variable: boundary conditions}
            Boundary conditions of the model

        Returns
        -------
        :class:`pybamm.BinaryOperator` (or subclass)
            Discretised binary operator

        """
        # Pre-process children
        left, right = bin_op.children
        new_left = self.process_symbol(left, y_slices, boundary_conditions)
        new_right = self.process_symbol(right, y_slices, boundary_conditions)
        # Post-processing to make sure discretised dimensions match
        # If neither child has gradients, or both children have gradients
        # no need to do any averaging
        if (
            left.has_gradient_and_not_divergence()
            == right.has_gradient_and_not_divergence()
        ):
            pass
        # If only left child has gradient, compute diffusivity for right child
        elif (
            left.has_gradient_and_not_divergence()
            and not right.has_gradient_and_not_divergence()
        ):
            new_right = self.compute_diffusivity(new_right)
        # If only right child has gradient, compute diffusivity for left child
        elif (
            right.has_gradient_and_not_divergence()
            and not left.has_gradient_and_not_divergence()
        ):
            new_left = self.compute_diffusivity(new_left)
        # Return new binary operator with appropriate class
        return bin_op.__class__(new_left, new_right)

    def compute_diffusivity(self, symbol):
        """Compute diffusivity; default behaviour is identity operator"""
        return symbol

    def gradient(self, symbol, y_slices, boundary_conditions):
        """How to discretise gradient operators.

        Parameters
        ----------
        symbol : :class:`Symbol` (or subclass)
            The symbol (typically a variable) of which to take the gradient
        y_slices : slice
            The slice to assign to StateVector when discretising a variable
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        """
        raise NotImplementedError

    def divergence(self, symbol, y_slices, boundary_conditions):
        """How to discretise divergence operators.

        Parameters
        ----------
        symbol : :class:`Symbol` (or subclass)
            The symbol (typically a variable) of which to take the divergence
        y_slices : slice
            The slice to assign to StateVector when discretising a variable
        boundary_conditions : dict
            The boundary conditions of the model

        """
        raise NotImplementedError

    def vector_of_ones(self, domain):
        """
        Returns a Vector of ones of the size given by the mesh.
        """

        mesh_points = np.array([])

        for dom in domain:
            mesh_points = np.concatenate([mesh_points, self.mesh[dom].nodes])
        return pybamm.Vector(np.ones_like(mesh_points))

    def concatenate(self, *symbols):
        return pybamm.NumpyConcatenation(*symbols)

    def check_model(self, model):
        """ Perform some basic checks to make sure the discretised model makes sense."""
        # Check initial conditions are a numpy array
        # Individual
        for var, eqn in model.initial_conditions.items():
            assert type(eqn.evaluate(0, None)) is np.ndarray, pybamm.ModelError(
                """
                initial_conditions must be numpy array after discretisation but they are
                {} for variable '{}'.
                """.format(
                    type(eqn.evaluate(0, None)), var
                )
            )
        # Concatenated
        assert (
            type(model.concatenated_initial_conditions) is np.ndarray
        ), pybamm.ModelError(
            """
            Concatenated initial_conditions must be numpy array after discretisation but
            they are {}.
            """.format(
                type(model.concatenated_initial_conditions)
            )
        )

        # Check initial conditions and rhs have the same shape
        y0 = model.concatenated_initial_conditions
        # Individual
        for var in model.rhs.keys():
            assert (
                model.rhs[var].evaluate(0, y0).shape
                == model.initial_conditions[var].evaluate(0, None).shape
            ), pybamm.ModelError(
                """
                rhs and initial_conditions must have the same shape after discretisation
                but rhs.shape = {} and initial_conditions.shape = {} for variable '{}'.
                """.format(
                    model.rhs[var].evaluate(0, y0).shape,
                    model.initial_conditions[var].evaluate(0, None).shape,
                    var,
                )
            )
        # Concatenated
        assert (
            model.concatenated_rhs.evaluate(0, y0).shape == y0.shape
        ), pybamm.ModelError(
            """
            Concatenated rhs and initial_conditions must have the same shape after
            discretisation but rhs.shape = {} and initial_conditions.shape = {}.
            """.format(
                model.concatenated_rhs.evaluate(0, y0).shape, y0.shape
            )
        )

        # Check variables in variable list against rhs
        for var in model.rhs.keys():
            if var.name in model.variables.keys():
                assert (
                    model.rhs[var].evaluate(0, y0).shape
                    == model.variables[var.name].evaluate(0, y0).shape
                ), pybamm.ModelError(
                    """
                    variable and its eqn must have the same shape after discretisation
                    but variable.shape = {} and rhs.shape = {} for variable '{}'.
                    """.format(
                        model.variables[var.name].evaluate(0, y0).shape,
                        model.rhs[var].evaluate(0, y0).shape,
                        var,
                    )
                )
