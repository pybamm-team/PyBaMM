#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import copy
import numpy as np


class BaseDiscretisation(object):
    """The base discretisation class, with methods to process a model and replace
    Spatial Operators with Matrices and Variables with StateVectors

    Parameters
    ----------
    mesh_type: class name
        the type of combined mesh being used (e.g. Mesh)
    submesh_pts : dict
        the number of points on each of the subdomains
    submesh_types : dict
        the types of submeshes being employed on each of the subdomains
    """

    def __init__(self, mesh_type, submesh_pts, submesh_types):
        self.mesh_type = mesh_type
        self.submesh_pts = submesh_pts
        self.submesh_types = submesh_types

        self._mesh = {}

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh_in):
        self._mesh = mesh_in

    def mesh_geometry(self, geometry):
        self._mesh = self.mesh_type(geometry, self.submesh_types, self.submesh_pts)

    def process_model(self, model, geometry):
        """Discretise a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        """

        # Set the mesh
        self._mesh = self.mesh_type(geometry, self.submesh_types, self.submesh_pts)

        # Set the y split for variables
        variables = self.get_all_variables(model)
        y_slices = self.get_variable_slices(variables)

        # Discretise initial conditions
        model.initial_conditions = self.process_dict(model.initial_conditions)
        model.initial_conditions_ydot = self.process_dict(model.initial_conditions_ydot)

        # Concatenate initial conditions into a single vector

        # check that all initial conditions are set
        model.concatenated_initial_conditions = self._concatenate_init(
            model.initial_conditions, y_slices
        ).evaluate(0, None)

        # evaluate initial conditions for ydot if they exist
        if len(model.initial_conditions_ydot) > 0:
            model.concatenated_initial_conditions_ydot = self._concatenate_init(
                model.initial_conditions_ydot, y_slices
            ).evaluate(0, None)
        else:
            model.concatenated_initial_conditions_ydot = np.array([])

        # Discretise right-hand sides, passing domain from variable
        model.rhs = self.process_dict(model.rhs, y_slices, model.boundary_conditions)
        # Concatenate rhs into a single state vector
        model.concatenated_rhs = self.concatenate(*model.rhs.values())

        # Discretise and concatenate algebraic equations
        model.algebraic = self.process_list(
            model.algebraic, y_slices, model.boundary_conditions
        )
        model.concatenated_algebraic = self.concatenate(*model.algebraic)

        # Discretise variables (applying boundary conditions)
        # Note that we **do not** discretise the keys of model.rhs,
        # model.initial_conditions and model.boundary_conditions
        model.variables = self.process_dict(
            model.variables, y_slices, model.boundary_conditions
        )
        self.check_model(model)

    def get_all_variables(self, model):
        """get all variables in a model

        This returns all the variables in the model in a list. Given a model with n
        differential equations in model.rhs, the returned list will have the
        corresponding variables for these equations (in order) in the first n elements.
        The remaining elements of the list will have the remaining algebraic variables
        in no specific order

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            Model to dicretise.


        Returns
        -------
        variables: list of :class:`pybamm.Variable`
            All the variables in the model
        """
        variables = list(model.rhs.keys())
        algebraic_variables = {}
        for equation in model.algebraic:
            # find all variables in the expression
            eq_variables = {
                variable.id: pybamm.Variable(variable.name, variable.domain)
                for variable in equation.pre_order()
                if isinstance(variable, pybamm.Variable)
            }

            # add to the total set
            algebraic_variables.update(eq_variables)

        # remove rhs variables from algebraic variables (if they exist)
        for variable in variables:
            algebraic_variables.pop(variable.id, None)

        # return both rhs and algebraic variables
        return variables + list(algebraic_variables.values())

    def get_variable_slices(self, variables):
        """Set the slicing for variables.

        Parameters
        ----------
        variables : list of Variables
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
            # If domain is empty then variable has size 1
            if variable.domain == []:
                end += 1
            # Otherwise, add up the size of all the domains in variable.domain
            else:
                for dom in variable.domain:
                    end += self.mesh[dom].npts
            y_slices[variable.id] = slice(start, end)
            start = end

        return y_slices

    def process_dict(self, var_eqn_dict, y_slices=None, boundary_conditions={}):
        """Discretise a dictionary of {variable: equation}, broadcasting if necessary
        (can be model.rhs, model.initial_conditions or model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
            (can be model.rhs, model.initial_conditions or model.variables)
        y_slices : dict of {variable id: slice}
            The slices to assign to StateVectors when discretising
        boundary_conditions : dict
            Boundary conditions ({symbol.id: {"left": left bc, "right": right bc}} dict)
            associated with var_eqn_dict to dicretise

        Returns
        -------
        var_eqn_dict : dict
            Discretised equations

        """
        # reformat boundary conditions (for rhs and variables)
        boundary_conditions = {
            key.id: value for key, value in boundary_conditions.items()
        }
        for variable, equation in var_eqn_dict.items():
            if equation.evaluates_to_number():
                # Broadcast scalar equation to the domain specified by variable.domain
                equation = self.broadcast(equation, variable.domain)

            # Process symbol (original or broadcasted)
            var_eqn_dict[variable] = self.process_symbol(
                equation, y_slices, boundary_conditions
            )

        return var_eqn_dict

    def process_list(self, var_eqn_list, y_slices, boundary_conditions):
        """Discretise a list of [equation]
        (typically model.algebraic).

        Parameters
        ----------
        var_eqn_list:list
            Equations to dicretise
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
        for i, equation in enumerate(var_eqn_list):
            var_eqn_list[i] = self.process_symbol(
                equation, y_slices, boundary_conditions
            )

        return var_eqn_list

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

        elif isinstance(symbol, pybamm.Divergence):
            return self.divergence(symbol.children[0], y_slices, boundary_conditions)

        elif isinstance(symbol, pybamm.Broadcast):
            # Process child first
            new_child = self.process_symbol(
                symbol.children[0], y_slices, boundary_conditions
            )
            # Broadcast new_child to the domain specified by symbol.domain
            # Different discretisations may broadcast differently
            return self.broadcast(new_child, symbol.domain)

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

        elif isinstance(symbol, pybamm.Space):
            symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
            return pybamm.Vector(symbol_mesh.nodes)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [
                self.process_symbol(child, y_slices, boundary_conditions)
                for child in symbol.children
            ]
            new_symbol = pybamm.DomainConcatenation(new_children, self.mesh)

            if new_symbol.is_constant():
                return pybamm.Vector(new_symbol.evaluate())

            return new_symbol

        else:
            return copy.deepcopy(symbol)

    def process_binary_operators(self, bin_op, y_slices, boundary_conditions):
        """Discretise binary operators in model equations.  Performs appropriate
        averaging of diffusivities if one of the children is a gradient operator, so
        that discretised sizes match up.  This is mainly an issue for the Finite Volume
        Discretisation: see
        :meth:`pybamm.FiniteVolumeDiscretisation.compute_diffusivity()`

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

    def broadcast(self, symbol, domain):
        """
        Broadcast symbol to a specified domain. To do this, calls
        :class:`pybamm.NumpyBroadcast`

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to be broadcasted
        domain : iterable of string
            The domain to broadcast to
        """
        # create broadcasted symbol
        broadcasted_symbol = pybamm.NumpyBroadcast(symbol, domain, self.mesh)

        # if the broadcasted symbol evaluates to a constant value, replace the
        # symbol-Vector multiplication with a single array
        if broadcasted_symbol.is_constant():
            broadcasted_symbol = pybamm.Array(
                broadcasted_symbol.evaluate(), domain=broadcasted_symbol.domain
            )

        return broadcasted_symbol

    def concatenate(self, *symbols):
        return pybamm.NumpyModelConcatenation(*symbols)

    def _concatenate_init(self, var_eqn_dict, y_slices):
        """
        Concatenate a dictionary of {variable: equation} initial conditions using
        y_slices

        The keys/variables in `var_eqn_dict` must be the same as the ids in `y_slices`
        The resultant concatenation is ordered according to the ordering of the slice
        values in `y_slices`

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
        y_slices : dict of {variable id: slice}
            The slices to assign to StateVectors when discretising

        Returns
        -------
        var_eqn_dict : dict
            Discretised right-hand side equations

        """
        ids = {v.id for v in var_eqn_dict.keys()}
        if ids != y_slices.keys():
            given_variable_names = [v.name for v in var_eqn_dict.keys()]
            raise pybamm.ModelError(
                "Initial conditions are insufficient. Only "
                "provided for {} ".format(given_variable_names)
            )

        equations = list(var_eqn_dict.values())
        slices = [y_slices[var.id] for var in var_eqn_dict.keys()]

        # sort equations according to slices
        sorted_equations = [eq for _, eq in sorted(zip(slices, equations))]

        return self.concatenate(*sorted_equations)

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
            model.concatenated_rhs.evaluate(0, y0).shape[0]
            + model.concatenated_algebraic.evaluate(0, y0).shape[0]
            == y0.shape[0]
        ), pybamm.ModelError(
            """
            Concatenation of (rhs, algebraic) and initial_conditions must have the
            same shape after discretisation but rhs.shape = {}, algebraic.shape = {},
            and initial_conditions.shape = {}.
            """.format(
                model.concatenated_rhs.evaluate(0, y0).shape,
                model.concatenated_algebraic.evaluate(0, y0).shape,
                y0.shape,
            )
        )

        # Check variables in variable list against rhs
        # Be lenient with size check if the variable in model.variables is broadcasted
        for var in model.rhs.keys():
            if var.name in model.variables.keys():
                assert model.rhs[var].evaluate(0, y0).shape == model.variables[
                    var.name
                ].evaluate(0, y0).shape or isinstance(
                    model.variables[var.name], pybamm.NumpyBroadcast
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
