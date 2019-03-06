#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import copy
import numpy as np


class Discretisation(object):
    """The discretisation class, with methods to process a model and replace
    Spatial Operators with Matrices and Variables with StateVectors

    Parameters
    ----------
    mesh : pybamm.Mesh
            contains all submeshes to be used on each domain
    spatial_methods : dict
            a dictionary of the spatial method to be used on each
            domain. The keys correspond to the keys in a pybamm.Model
    """

    def __init__(self, mesh, spatial_methods):
        self._mesh = mesh
        # Unpack macroscale to the constituent subdomains
        if "macroscale" in spatial_methods.keys():
            method = spatial_methods["macroscale"]
            spatial_methods["negative electrode"] = method
            spatial_methods["separator"] = method
            spatial_methods["positive electrode"] = method
        self._spatial_methods = {
            dom: method(mesh) for dom, method in spatial_methods.items()
        }
        self._bcs = {}
        self._y_slices = {}

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
        # set boundary conditions (only need key ids for boundary_conditions)
        self._bcs = {key.id: value for key, value in model.boundary_conditions.items()}
        # set variables (we require the full variable not just id)
        variables = list(model.rhs.keys()) + list(model.algebraic.keys())

        # Set the y split for variables
        self.set_variable_slices(variables)

        # Process initial condtions
        self.process_initial_conditions(model)

        # Process parabolic and elliptic equations
        self.process_rhs_and_algebraic(model)

        # Discretise variables (applying boundary conditions)
        # Note that we **do not** discretise the keys of model.rhs,
        # model.initial_conditions and model.boundary_conditions
        model.variables = self.process_dict(model.variables)

        # Check that resulting model makes sense
        self.check_model(model)

    def set_variable_slices(self, variables):
        """Sets the slicing for variables.

        variables : iterable of :class:`pybamm.Variables`
            The variables for which to set slices
        """
        # Unpack symbols in variables that are concatenations of variables
        unpacked_variables = []
        for symbol in variables:
            if isinstance(symbol, pybamm.Concatenation):
                unpacked_variables.extend([var for var in symbol.children])
            else:
                unpacked_variables.append(symbol)
        # Set up y_slices
        y_slices = {variable.id: None for variable in unpacked_variables}
        start = 0
        end = 0
        # Iterate through unpacked variables, adding appropriate slices to y_slices
        for variable in unpacked_variables:
            # If domain is empty then variable has size 1
            if variable.domain == []:
                end += 1
            # Otherwise, add up the size of all the domains in variable.domain
            else:
                for dom in variable.domain:
                    end += self._spatial_methods[dom].mesh[dom].npts_for_broadcast
            y_slices[variable.id] = slice(start, end)
            start = end
        self._y_slices = y_slices

        assert isinstance(self._y_slices, dict), ValueError(
            """y_slices should be dict, not {}""".format(type(self._y_slices))
        )

    def process_initial_conditions(self, model):
        """Discretise model initial_conditions.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})

        """
        # Discretise initial conditions
        model.initial_conditions = self.process_dict(model.initial_conditions)
        model.initial_conditions_ydot = self.process_dict(model.initial_conditions_ydot)

        # Concatenate initial conditions into a single vector
        # check that all initial conditions are set
        model.concatenated_initial_conditions = self._concatenate_init(
            model.initial_conditions
        ).evaluate(0, None)

        # evaluate initial conditions for ydot if they exist
        if len(model.initial_conditions_ydot) > 0:
            model.concatenated_initial_conditions_ydot = self._concatenate_init(
                model.initial_conditions_ydot
            ).evaluate(0, None)
        else:
            model.concatenated_initial_conditions_ydot = np.array([])

    def process_rhs_and_algebraic(self, model):
        """Discretise model equations - differential ('rhs') and algebraic.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            Model to dicretise. Must have attributes rhs, initial_conditions and
            boundary_conditions (all dicts of {variable: equation})
        """
        # Discretise right-hand sides, passing domain from variable
        model.rhs = self.process_dict(model.rhs)
        # Concatenate rhs into a single state vector
        model.concatenated_rhs = self.concatenate(*model.rhs.values())

        # Discretise and concatenate algebraic equations
        model.algebraic = self.process_dict(model.algebraic)
        model.concatenated_algebraic = self.concatenate(*model.algebraic.values())

    def process_dict(self, var_eqn_dict):
        """Discretise a dictionary of {variable: equation}, broadcasting if necessary
        (can be model.rhs, model.initial_conditions or model.variables).

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise
            (can be model.rhs, model.initial_conditions or model.variables)

        Returns
        -------
        var_eqn_dict : dict
            Discretised equations

        """

        for eqn_key, eqn in var_eqn_dict.items():
            # Broadcast if the equation evaluates to a number(e.g. Scalar)
            if eqn.evaluates_to_number():
                if isinstance(eqn_key, str):
                    eqn = pybamm.Broadcast(eqn, [])
                elif eqn_key.domain == []:
                    eqn = pybamm.Broadcast(eqn, eqn_key.domain)
                else:
                    eqn = self._spatial_methods[eqn_key.domain[0]].broadcast(
                        eqn, eqn_key.domain
                    )

            # Process symbol (original or broadcasted)
            var_eqn_dict[eqn_key] = self.process_symbol(eqn)
            # note we are sending in the key.id here so we don't have to
            # keep calling .id
        return var_eqn_dict

    def process_symbol(self, symbol):
        """Discretise operators in model equations.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol to discretise

        Returns
        -------
        :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Discretised symbol

        """
        if isinstance(symbol, pybamm.Gradient):
            child = symbol.children[0]
            discretised_child = self.process_symbol(child)
            return self._spatial_methods[symbol.domain[0]].gradient(
                child, discretised_child, self._bcs
            )

        elif isinstance(symbol, pybamm.Divergence):
            child = symbol.children[0]
            discretised_child = self.process_symbol(child)
            return self._spatial_methods[symbol.domain[0]].divergence(
                child, discretised_child, self._bcs
            )

        elif isinstance(symbol, pybamm.Integral):
            return self.integral(
                symbol.children[0], y_slices, boundary_conditions, symbol.definite
            )

        elif isinstance(symbol, pybamm.Broadcast):
            # Process child first
            new_child = self.process_symbol(symbol.children[0])
            # Broadcast new_child to the domain specified by symbol.domain
            # Different discretisations may broadcast differently
            if symbol.domain == []:
                symbol = pybamm.NumpyBroadcast(new_child, symbol.domain, {})
            else:
                symbol = self._spatial_methods[symbol.domain[0]].broadcast(
                    new_child, symbol.domain
                )
            return symbol

        elif isinstance(symbol, pybamm.SurfaceValue):
            child = symbol.children[0]
            discretised_child = self.process_symbol(child)
            return self._spatial_methods[child.domain[0]].surface_value(
                discretised_child
            )

        elif isinstance(symbol, pybamm.BinaryOperator):
            return self.process_binary_operators(symbol)

        elif isinstance(symbol, pybamm.Function):
            new_child = self.process_symbol(symbol.children[0])
            return pybamm.Function(symbol.func, new_child)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.children[0])
            return symbol.__class__(new_child)

        elif isinstance(symbol, pybamm.Variable):
            return pybamm.StateVector(self._y_slices[symbol.id], domain=symbol.domain)

        elif isinstance(symbol, pybamm.SpatialVariable):
            return self._spatial_methods[symbol.domain[0]].spatial_variable(symbol)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            new_symbol = pybamm.DomainConcatenation(new_children, self.mesh)

            if new_symbol.is_constant():
                value = new_symbol.evaluate()
                return pybamm.Vector(value)
            return new_symbol

        else:
            new_symbol = copy.deepcopy(symbol)
            new_symbol.parent = None
            return new_symbol

    def process_binary_operators(self, bin_op):
        """Discretise binary operators in model equations.  Performs appropriate
        averaging of diffusivities if one of the children is a gradient operator, so
        that discretised sizes match up.  This is mainly an issue for the Finite Volume
        Discretisation: see
        :meth:`pybamm.FiniteVolumeDiscretisation.compute_diffusivity()`

        Parameters
        ----------
        bin_op : :class:`pybamm.BinaryOperator` (or subclass)
            Binary operator to discretise

        Returns
        -------
        :class:`pybamm.BinaryOperator` (or subclass)
            Discretised binary operator

        """
        # Pre-process children
        left, right = bin_op.children
        new_left = self.process_symbol(left)
        new_right = self.process_symbol(right)
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
            new_right = self._spatial_methods[bin_op.domain[0]].compute_diffusivity(
                new_right
            )
        # If only right child has gradient, compute diffusivity for left child
        elif (
            right.has_gradient_and_not_divergence()
            and not left.has_gradient_and_not_divergence()
        ):
            new_left = self._spatial_methods[bin_op.domain[0]].compute_diffusivity(
                new_left
            )
        # Return new binary operator with appropriate class
        return bin_op.__class__(new_left, new_right)

    def concatenate(self, *symbols):
        return pybamm.NumpyConcatenation(*symbols)

    def _concatenate_init(self, var_eqn_dict):
        """
        Concatenate a dictionary of {variable: equation} initial conditions using
        self._y_slices

        The keys/variables in `var_eqn_dict` must be the same as the ids in
        `self._y_slices`.
        The resultant concatenation is ordered according to the ordering of the slice
        values in `self._y_slices`

        Parameters
        ----------
        var_eqn_dict : dict
            Equations ({variable: equation} dict) to dicretise

                Returns
        -------
        var_eqn_dict : dict
            Discretised right-hand side equations

        """
        # Unpack symbols in variables that are concatenations of variables
        unpacked_variables = []
        for symbol in var_eqn_dict.keys():
            if isinstance(symbol, pybamm.Concatenation):
                unpacked_variables.extend([var for var in symbol.children])
            else:
                unpacked_variables.append(symbol)
        # Check keys from the given var_eqn_dict against self._y_slices
        ids = {v.id for v in unpacked_variables}
        if ids != self._y_slices.keys():
            given_variable_names = [v.name for v in var_eqn_dict.keys()]
            raise pybamm.ModelError(
                "Initial conditions are insufficient. Only "
                "provided for {} ".format(given_variable_names)
            )

        equations = list(var_eqn_dict.values())
        slices = [self._y_slices[var.id] for var in unpacked_variables]

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
