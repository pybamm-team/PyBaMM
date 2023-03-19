#
# Base model class
#
import numbers
from collections import OrderedDict

import copy
import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.operations.latexify import Latexify


class BaseModel:
    """
    Base model class for other models to extend.

    Attributes
    ----------
    name: str
        A string giving the name of the model.
    options: dict
        A dictionary of options to be passed to the model.
    submodels: dict
        A dictionary of submodels that the model is composed of.
    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs.
    algebraic: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the algebraic equations. The algebraic expressions are assumed to equate
        to zero. Note that all the variables in the model must exist in the keys of
        `rhs` or `algebraic`.
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions for the state variables y. The initial conditions for
        algebraic variables are provided as initial guesses to a root finding algorithm
        that calculates consistent initial conditions.
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions.
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables.
    events: list of :class:`pybamm.Event`
        A list of events. Each event can either cause the solver to terminate
        (e.g. concentration goes negative), or be used to inform the solver of the
        existance of a discontinuity (e.g. discontinuity in the input current).
    concatenated_rhs : :class:`pybamm.Concatenation`
        After discretisation, contains the expressions representing the rhs equations
        concatenated into a single expression.
    concatenated_algebraic : :class:`pybamm.Concatenation`
        After discretisation, contains the expressions representing the algebraic
        equations concatenated into a single expression.
    concatenated_initial_conditions : :class:`numpy.array`
        After discretisation, contains the vector of initial conditions.
    mass_matrix : :class:`pybamm.Matrix`
        After discretisation, contains the mass matrix for the model. This is computed
        automatically.
    mass_matrix_inv : :class:`pybamm.Matrix`
        After discretisation, contains the inverse mass matrix for the differential
        (rhs) part of model. This is computed automatically.
    jacobian : :class:`pybamm.Concatenation`
        Contains the Jacobian for the model. If model.use_jacobian is True, the
        Jacobian is computed automatically during solver set up.
    jacobian_rhs : :class:`pybamm.Concatenation`
        Contains the Jacobian for the part of the model which contains time derivatives.
        If model.use_jacobian is True, the Jacobian is computed automatically during
        solver set up.
    jacobian_algebraic : :class:`pybamm.Concatenation`
        Contains the Jacobian for the algebraic part of the model. This may be used
        by the solver when calculating consistent initial conditions. If
        model.use_jacobian is True, the Jacobian is computed automatically during
        solver set up.
    use_jacobian : bool
        Whether to use the Jacobian when solving the model (default is True).
    convert_to_format : str
        Whether to convert the expression trees representing the rhs and
        algebraic equations, Jacobain (if using) and events into a different format:

        - None: keep PyBaMM expression tree structure.
        - "python": convert into pure python code that will calculate the result of \
        calling `evaluate(t, y)` on the given expression treeself.
        - "casadi": convert into CasADi expression tree, which then uses CasADi's \
        algorithm to calculate the Jacobian.

        Default is "casadi".
    """

    def __init__(self, name="Unnamed model"):
        self.name = name
        self._options = {}
        self._built = False
        self._built_fundamental = False

        # Initialise empty model
        self.submodels = {}
        self._rhs = {}
        self._algebraic = {}
        self._initial_conditions = {}
        self._boundary_conditions = {}
        self._variables = pybamm.FuzzyDict({})
        self._events = []
        self._concatenated_rhs = None
        self._concatenated_algebraic = None
        self._concatenated_initial_conditions = None
        self._mass_matrix = None
        self._mass_matrix_inv = None
        self._jacobian = None
        self._jacobian_algebraic = None
        self._parameters = None
        self._input_parameters = None
        self._parameter_info = None
        self._variables_casadi = {}

        # Default behaviour is to use the jacobian
        self.use_jacobian = True
        self.convert_to_format = "casadi"

        # Model is not initially discretised
        self.is_discretised = False
        self.y_slices = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, rhs):
        self._rhs = EquationDict("rhs", rhs)

    @property
    def algebraic(self):
        return self._algebraic

    @algebraic.setter
    def algebraic(self, algebraic):
        self._algebraic = EquationDict("algebraic", algebraic)

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        self._initial_conditions = EquationDict(
            "initial_conditions", initial_conditions
        )

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        self._boundary_conditions = BoundaryConditionsDict(boundary_conditions)

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        for name, var in variables.items():
            if (
                isinstance(var, pybamm.Variable)
                and var.name != name
                # Exception if the variable is also there under its own name
                and not (var.name in variables and variables[var.name] == var)
            ):
                raise ValueError(
                    f"Variable with name '{var.name}' is in variables dictionary with "
                    f"name '{name}'. Names must match."
                )
        self._variables = pybamm.FuzzyDict(variables)

    def variable_names(self):
        return list(self._variables.keys())

    @property
    def variables_and_events(self):
        """
        Returns variables and events in a single dictionary
        """
        try:
            return self._variables_and_events
        except AttributeError:
            self._variables_and_events = self.variables.copy()
            self._variables_and_events.update(
                {f"Event: {event.name}": event.expression for event in self.events}
            )
            return self._variables_and_events

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events):
        self._events = events

    @property
    def concatenated_rhs(self):
        return self._concatenated_rhs

    @concatenated_rhs.setter
    def concatenated_rhs(self, concatenated_rhs):
        self._concatenated_rhs = concatenated_rhs

    @property
    def concatenated_algebraic(self):
        return self._concatenated_algebraic

    @concatenated_algebraic.setter
    def concatenated_algebraic(self, concatenated_algebraic):
        self._concatenated_algebraic = concatenated_algebraic

    @property
    def concatenated_initial_conditions(self):
        return self._concatenated_initial_conditions

    @concatenated_initial_conditions.setter
    def concatenated_initial_conditions(self, concatenated_initial_conditions):
        self._concatenated_initial_conditions = concatenated_initial_conditions

    @property
    def mass_matrix(self):
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self, mass_matrix):
        self._mass_matrix = mass_matrix

    @property
    def mass_matrix_inv(self):
        return self._mass_matrix_inv

    @mass_matrix_inv.setter
    def mass_matrix_inv(self, mass_matrix_inv):
        self._mass_matrix_inv = mass_matrix_inv

    @property
    def jacobian(self):
        return self._jacobian

    @jacobian.setter
    def jacobian(self, jacobian):
        self._jacobian = jacobian

    @property
    def jacobian_rhs(self):
        return self._jacobian_rhs

    @jacobian_rhs.setter
    def jacobian_rhs(self, jacobian_rhs):
        self._jacobian_rhs = jacobian_rhs

    @property
    def jacobian_algebraic(self):
        return self._jacobian_algebraic

    @jacobian_algebraic.setter
    def jacobian_algebraic(self, jacobian_algebraic):
        self._jacobian_algebraic = jacobian_algebraic

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, values):
        self._param = values

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        self._options = options

    @property
    def timescale(self):
        raise NotImplementedError(
            "timescale has been removed since models are now dimensional"
        )

    @timescale.setter
    def timescale(self, value):
        raise NotImplementedError(
            "timescale has been removed since models are now dimensional"
        )

    @property
    def length_scales(self):
        raise NotImplementedError(
            "length_scales has been removed since models are now dimensional"
        )

    @length_scales.setter
    def length_scales(self, values):
        raise NotImplementedError(
            "length_scales has been removed since models are now dimensional"
        )

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_geometry(self):
        return {}

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_solver(self):
        """Return default solver based on whether model is ODE/DAE or algebraic"""
        if len(self.rhs) == 0 and len(self.algebraic) != 0:
            return pybamm.CasadiAlgebraicSolver()
        else:
            return pybamm.CasadiSolver(mode="safe")

    @property
    def default_quick_plot_variables(self):
        return None

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({})

    @property
    def parameters(self):
        """Returns all the parameters in the model"""
        self._parameters = self._find_symbols(
            (pybamm.Parameter, pybamm.InputParameter, pybamm.FunctionParameter)
        )
        return self._parameters

    @property
    def input_parameters(self):
        """Returns all the input parameters in the model"""
        if self._input_parameters is None:
            self._input_parameters = self._find_symbols(pybamm.InputParameter)
        return self._input_parameters

    def print_parameter_info(self):
        self._parameter_info = ""
        parameters = self._find_symbols(pybamm.Parameter)
        for param in parameters:
            self._parameter_info += f"{param.name} (Parameter)\n"
        input_parameters = self._find_symbols(pybamm.InputParameter)
        for input_param in input_parameters:
            if input_param.domain == []:
                self._parameter_info += f"{input_param.name} (InputParameter)\n"
            else:
                self._parameter_info += (
                    f"{input_param.name} (InputParameter in {input_param.domain})\n"
                )
        function_parameters = self._find_symbols(pybamm.FunctionParameter)
        for func_param in function_parameters:
            # don't double count function parameters
            if func_param.name not in self._parameter_info:
                input_names = "'" + "', '".join(func_param.input_names) + "'"
                self._parameter_info += (
                    f"{func_param.name} (FunctionParameter "
                    f"with input(s) {input_names})\n"
                )

        print(self._parameter_info)

    def _find_symbols(self, typ):
        """Find all the instances of `typ` in the model"""
        unpacker = pybamm.SymbolUnpacker(typ)
        all_input_parameters = unpacker.unpack_list_of_symbols(
            list(self.rhs.values())
            + list(self.algebraic.values())
            + list(self.initial_conditions.values())
            + [
                x[side][0]
                for x in self.boundary_conditions.values()
                for side in x.keys()
            ]
            + list(self.variables.values())
            + [event.expression for event in self.events]
        )
        return list(all_input_parameters)

    def new_copy(self):
        """
        Creates a copy of the model, explicitly copying all the mutable attributes
        to avoid issues with shared objects.
        """
        new_model = copy.copy(self)
        new_model._rhs = self.rhs.copy()
        new_model._algebraic = self.algebraic.copy()
        new_model._initial_conditions = self.initial_conditions.copy()
        new_model._boundary_conditions = self.boundary_conditions.copy()
        new_model._variables = self.variables.copy()
        new_model._events = self.events.copy()
        new_model._variables_casadi = self._variables_casadi.copy()
        return new_model

    def update(self, *submodels):
        """
        Update model to add new physics from submodels

        Parameters
        ----------
        submodel : iterable of :class:`pybamm.BaseModel`
            The submodels from which to create new model
        """
        for submodel in submodels:
            # check and then update dicts
            self.check_and_combine_dict(self._rhs, submodel.rhs)
            self.check_and_combine_dict(self._algebraic, submodel.algebraic)
            self.check_and_combine_dict(
                self._initial_conditions, submodel.initial_conditions
            )
            self.check_and_combine_dict(
                self._boundary_conditions, submodel.boundary_conditions
            )
            self.variables.update(submodel.variables)  # keys are strings so no check
            self._events += submodel.events

    def build_fundamental(self):
        # Get the fundamental variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        self._built_fundamental = True

    def build_coupled_variables(self):
        # Note: pybamm will try to get the coupled variables for the submodels in the
        # order they are set by the user. If this fails for a particular submodel,
        # return to it later and try again. If setting coupled variables fails and
        # there are no more submodels to try, raise an error.
        submodels = list(self.submodels.keys())
        count = 0
        # For this part the FuzzyDict of variables is briefly converted back into a
        # normal dictionary for speed with KeyErrors
        self._variables = dict(self._variables)
        while len(submodels) > 0:
            count += 1
            for submodel_name, submodel in self.submodels.items():
                if submodel_name in submodels:
                    pybamm.logger.debug(
                        "Getting coupled variables for {} submodel ({})".format(
                            submodel_name, self.name
                        )
                    )
                    try:
                        self.variables.update(
                            submodel.get_coupled_variables(self.variables)
                        )
                        submodels.remove(submodel_name)
                    except KeyError as key:
                        if len(submodels) == 1 or count == 100:
                            # no more submodels to try
                            raise pybamm.ModelError(
                                "Missing variable for submodel '{}': {}.\n".format(
                                    submodel_name, key
                                )
                                + "Check the selected "
                                "submodels provide all of the required variables."
                            )
                        else:
                            # try setting coupled variables on next loop through
                            pybamm.logger.debug(
                                "Can't find {}, trying other submodels first".format(
                                    key
                                )
                            )
        # Convert variables back into FuzzyDict
        self.variables = pybamm.FuzzyDict(self._variables)

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.verbose(
                "Setting rhs for {} submodel ({})".format(submodel_name, self.name)
            )

            submodel.set_rhs(self.variables)
            pybamm.logger.verbose(
                "Setting algebraic for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )

            submodel.set_algebraic(self.variables)
            pybamm.logger.verbose(
                "Setting boundary conditions for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )

            submodel.set_boundary_conditions(self.variables)
            pybamm.logger.verbose(
                "Setting initial conditions for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            submodel.set_initial_conditions(self.variables)
            submodel.set_events(self.variables)
            pybamm.logger.verbose(
                "Updating {} submodel ({})".format(submodel_name, self.name)
            )
            self.update(submodel)
            self.check_no_repeated_keys()

    def build_model(self):
        self._build_model()
        self._built = True
        pybamm.logger.info("Finish building {}".format(self.name))

    def _build_model(self):
        # Check if already built
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info("Start building {}".format(self.name))

        if self._built_fundamental is False:
            self.build_fundamental()

        self.build_coupled_variables()

        self.build_model_equations()

    def set_initial_conditions_from(self, solution, inplace=True, return_type="model"):
        """
        Update initial conditions with the final states from a Solution object or from
        a dictionary.
        This assumes that, for each variable in self.initial_conditions, there is a
        corresponding variable in the solution with the same name and size.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`, or dict
            The solution to use to initialize the model
        inplace : bool, optional
            Whether to modify the model inplace or create a new model (default True)
        return_type : str, optional
            Whether to return the model (default) or initial conditions ("ics")
        """
        initial_conditions = {}
        if isinstance(solution, pybamm.Solution):
            solution = solution.last_state
        for var in self.initial_conditions:
            if isinstance(var, pybamm.Variable):
                try:
                    final_state = solution[var.name]
                except KeyError as e:
                    raise pybamm.ModelError(
                        "To update a model from a solution, each variable in "
                        "model.initial_conditions must appear in the solution with "
                        "the same key as the variable name. In the solution provided, "
                        f"'{e.args[0]}' was not found."
                    )
                if isinstance(solution, pybamm.Solution):
                    final_state = final_state.data
                if final_state.ndim == 0:
                    final_state_eval = np.array([final_state])
                elif final_state.ndim == 1:
                    final_state_eval = final_state[-1:]
                elif final_state.ndim == 2:
                    final_state_eval = final_state[:, -1]
                elif final_state.ndim == 3:
                    final_state_eval = final_state[:, :, -1].flatten(order="F")
                else:
                    raise NotImplementedError("Variable must be 0D, 1D, or 2D")
            elif isinstance(var, pybamm.Concatenation):
                children = []
                for child in var.orphans:
                    try:
                        final_state = solution[child.name]
                    except KeyError as e:
                        raise pybamm.ModelError(
                            "To update a model from a solution, each variable in "
                            "model.initial_conditions must appear in the solution with "
                            "the same key as the variable name. In the solution "
                            f"provided, {e.args[0]}"
                        )
                    if isinstance(solution, pybamm.Solution):
                        final_state = final_state.data
                    if final_state.ndim == 2:
                        final_state_eval = final_state[:, -1]
                    else:
                        raise NotImplementedError(
                            "Variable in concatenation must be 1D"
                        )
                    children.append(final_state_eval)
                final_state_eval = np.concatenate(children)
            else:
                raise NotImplementedError(
                    "Variable must have type 'Variable' or 'Concatenation'"
                )

            # If the model is already discretised, then the initial conditions must
            # be scaled and offset (otherwise, this is done when the model is
            # discretised)
            if self.is_discretised:
                scale, reference = var.scale, var.reference
            else:
                scale, reference = 1, 0
            initial_conditions[var] = (
                pybamm.Vector(final_state_eval) - reference
            ) / scale

        # Also update the concatenated initial conditions if the model is already
        # discretised
        if self.is_discretised:
            # Unpack slices for sorting
            y_slices = {var: slce for var, slce in self.y_slices.items()}
            slices = []
            for symbol in self.initial_conditions.keys():
                if isinstance(symbol, pybamm.Concatenation):
                    # must append the slice for the whole concatenation, so that
                    # equations get sorted correctly
                    slices.append(
                        slice(
                            y_slices[symbol.children[0]][0].start,
                            y_slices[symbol.children[-1]][0].stop,
                        )
                    )
                else:
                    slices.append(y_slices[symbol][0])
            equations = list(initial_conditions.values())
            # sort equations according to slices
            sorted_equations = [eq for _, eq in sorted(zip(slices, equations))]
            concatenated_initial_conditions = pybamm.NumpyConcatenation(
                *sorted_equations
            )
        else:
            concatenated_initial_conditions = None

        if return_type == "model":
            if inplace is True:
                model = self
            else:
                model = self.new_copy()

            model.initial_conditions = initial_conditions
            model.concatenated_initial_conditions = concatenated_initial_conditions
            return model
        elif return_type == "ics":
            return initial_conditions, concatenated_initial_conditions

    def check_and_combine_dict(self, dict1, dict2):
        # check that the key ids are distinct
        ids1 = set(x for x in dict1.keys())
        ids2 = set(x for x in dict2.keys())
        if len(ids1.intersection(ids2)) != 0:
            variables = ids1.intersection(ids2)
            raise pybamm.ModelError(
                "Submodel incompatible: duplicate variables '{}'".format(variables)
            )
        dict1.update(dict2)

    def check_well_posedness(self, post_discretisation=False):
        """
        Check that the model is well-posed by executing the following tests:
        - Model is not over- or underdetermined, by comparing keys and equations in rhs
        and algebraic. Overdetermined if more equations than variables, underdetermined
        if more variables than equations.
        - There is an initial condition in self.initial_conditions for each
        variable/equation pair in self.rhs
        - There are appropriate boundary conditions in self.boundary_conditions for each
        variable/equation pair in self.rhs and self.algebraic

        Parameters
        ----------
        post_discretisation : boolean
            A flag indicating tests to be skipped after discretisation
        """
        self.check_for_time_derivatives()
        self.check_well_determined(post_discretisation)
        self.check_algebraic_equations(post_discretisation)
        self.check_ics_bcs()
        self.check_no_repeated_keys()
        # Can't check variables after discretising, since Variable objects get replaced
        # by StateVector objects
        # Checking variables is slow, so only do it in debug mode
        if pybamm.settings.debug_mode is True and post_discretisation is False:
            self.check_variables()

    def check_for_time_derivatives(self):
        # Check that no variable time derivatives exist in the rhs equations
        for key, eq in self.rhs.items():
            for node in eq.pre_order():
                if isinstance(node, pybamm.VariableDot):
                    raise pybamm.ModelError(
                        "time derivative of variable found "
                        "({}) in rhs equation {}".format(node, key)
                    )
                if isinstance(node, pybamm.StateVectorDot):
                    raise pybamm.ModelError(
                        "time derivative of state vector found "
                        "({}) in rhs equation {}".format(node, key)
                    )

        # Check that no variable time derivatives exist in the algebraic equations
        for key, eq in self.algebraic.items():
            for node in eq.pre_order():
                if isinstance(node, pybamm.VariableDot):
                    raise pybamm.ModelError(
                        "time derivative of variable found ({}) in algebraic"
                        "equation {}".format(node, key)
                    )
                if isinstance(node, pybamm.StateVectorDot):
                    raise pybamm.ModelError(
                        "time derivative of state vector found ({}) in algebraic"
                        "equation {}".format(node, key)
                    )

    def check_well_determined(self, post_discretisation):
        """Check that the model is not under- or over-determined."""
        # Equations (differential and algebraic)
        # Get all the variables from differential and algebraic equations
        all_vars_in_rhs_keys = set()
        all_vars_in_algebraic_keys = set()
        all_vars_in_eqns = set()
        # Get all variables ids from rhs and algebraic keys and equations, and
        # from boundary conditions
        # For equations we look through the whole expression tree.
        # "Variables" can be Concatenations so we also have to look in the whole
        # expression tree
        unpacker = pybamm.SymbolUnpacker((pybamm.Variable, pybamm.VariableDot))

        for var, eqn in self.rhs.items():
            # Find all variables and variabledot objects
            vars_in_rhs_keys = unpacker.unpack_symbol(var)
            vars_in_eqns = unpacker.unpack_symbol(eqn)

            # Look only for Variable (not VariableDot) in rhs keys
            all_vars_in_rhs_keys.update(
                [var for var in vars_in_rhs_keys if isinstance(var, pybamm.Variable)]
            )
            all_vars_in_eqns.update(vars_in_eqns)
        for var, eqn in self.algebraic.items():
            # Find all variables and variabledot objects
            vars_in_algebraic_keys = unpacker.unpack_symbol(var)
            vars_in_eqns = unpacker.unpack_symbol(eqn)

            # Store ids only
            # Look only for Variable (not VariableDot) in algebraic keys
            all_vars_in_algebraic_keys.update(
                [
                    var
                    for var in vars_in_algebraic_keys
                    if isinstance(var, pybamm.Variable)
                ]
            )
            all_vars_in_eqns.update(vars_in_eqns)
        for var, side_eqn in self.boundary_conditions.items():
            for side, (eqn, typ) in side_eqn.items():
                vars_in_eqns = unpacker.unpack_symbol(eqn)
                all_vars_in_eqns.update(vars_in_eqns)

        # If any keys are repeated between rhs and algebraic then the model is
        # overdetermined
        if not set(all_vars_in_rhs_keys).isdisjoint(all_vars_in_algebraic_keys):
            raise pybamm.ModelError("model is overdetermined (repeated keys)")
        # If any algebraic keys don't appear in the eqns (or bcs) then the model is
        # overdetermined (but rhs keys can be absent from the eqns, e.g. dcdt = -1 is
        # fine)
        # Skip this step after discretisation, as any variables in the equations will
        # have been discretised to slices but keys will still be variables
        extra_algebraic_keys = all_vars_in_algebraic_keys.difference(all_vars_in_eqns)
        if extra_algebraic_keys and not post_discretisation:
            raise pybamm.ModelError("model is overdetermined (extra algebraic keys)")
        # If any variables in the equations don't appear in the keys then the model is
        # underdetermined
        all_vars_in_keys = all_vars_in_rhs_keys.union(all_vars_in_algebraic_keys)
        extra_variables_in_equations = all_vars_in_eqns.difference(all_vars_in_keys)

        if extra_variables_in_equations:
            raise pybamm.ModelError("model is underdetermined (too many variables)")

    def check_algebraic_equations(self, post_discretisation):
        """
        Check that the algebraic equations are well-posed. After discretisation,
        there must be at least one StateVector in each algebraic equation.
        """
        if post_discretisation:
            # Check that each algebraic equation contains some StateVector
            for eqn in self.algebraic.values():
                if not eqn.has_symbol_of_classes(pybamm.StateVector):
                    raise pybamm.ModelError(
                        "each algebraic equation must contain at least one StateVector"
                    )
        else:
            # We do not perfom any checks before discretisation (most problematic
            # cases should be caught by `check_well_determined`)
            pass

    def check_ics_bcs(self):
        """Check that the initial and boundary conditions are well-posed."""
        # Initial conditions
        for var in self.rhs.keys():
            if var not in self.initial_conditions.keys():
                raise pybamm.ModelError(
                    """no initial condition given for variable '{}'""".format(var)
                )

    def check_variables(self):
        # Create list of all Variable nodes that appear in the model's list of variables
        unpacker = pybamm.SymbolUnpacker(pybamm.Variable)
        all_vars = unpacker.unpack_list_of_symbols(self.variables.values())

        vars_in_keys = set()

        model_keys = list(self.rhs.keys()) + list(self.algebraic.keys())

        for var in model_keys:
            if isinstance(var, pybamm.Variable):
                vars_in_keys.add(var)
            # Key can be a concatenation
            elif isinstance(var, pybamm.Concatenation):
                vars_in_keys.update(var.children)

        for var in all_vars:
            if var not in vars_in_keys:
                raise pybamm.ModelError(
                    """
                    No key set for variable '{}'. Make sure it is included in either
                    model.rhs or model.algebraic, in an unmodified form
                    (e.g. not Broadcasted)
                    """.format(
                        var
                    )
                )

    def check_no_repeated_keys(self):
        """Check that no equation keys are repeated."""
        rhs_keys = set(self.rhs.keys())
        alg_keys = set(self.algebraic.keys())

        if not rhs_keys.isdisjoint(alg_keys):
            raise pybamm.ModelError(
                "Multiple equations specified for variables {}".format(
                    rhs_keys.intersection(alg_keys)
                )
            )

    def info(self, symbol_name):
        """
        Provides helpful summary information for a symbol.

        Parameters
        ----------
        parameter_name : str
        """
        # Should we deprecate this? Not really sure how it's used?

        div = "-----------------------------------------"
        symbol = find_symbol_in_model(self, symbol_name)

        if symbol is None:
            return None

        print(div)
        print(symbol_name, "\n")
        print(type(symbol))

        if isinstance(symbol, pybamm.FunctionParameter):
            print("")
            print("Inputs:")
            symbol.print_input_names()

        print(div)

    def check_discretised_or_discretise_inplace_if_0D(self):
        """
        Discretise model if it isn't already discretised
        This only works with purely 0D models, as otherwise the mesh and spatial
        method should be specified by the user
        """
        if self.is_discretised is False:
            try:
                disc = pybamm.Discretisation()
                disc.process_model(self)
            except pybamm.DiscretisationError as e:
                raise pybamm.DiscretisationError(
                    "Cannot automatically discretise model, model should be "
                    "discretised before exporting casadi functions ({})".format(e)
                )

    def export_casadi_objects(self, variable_names, input_parameter_order=None):
        """
        Export the constituent parts of the model (rhs, algebraic, initial conditions,
        etc) as casadi objects.

        Parameters
        ----------
        variable_names : list
            Variables to be exported alongside the model structure
        input_parameter_order : list, optional
            Order in which the input parameters should be stacked.
            If input_parameter_order=None and len(self.input_parameters) > 1, a
            ValueError is raised (this helps to avoid accidentally using the wrong
            order)

        Returns
        -------
        casadi_dict : dict
            Dictionary of {str: casadi object} pairs representing the model in casadi
            format
        """
        self.check_discretised_or_discretise_inplace_if_0D()

        # Create casadi functions for the model
        t_casadi = casadi.MX.sym("t")
        y_diff = casadi.MX.sym("y_diff", self.concatenated_rhs.size)
        y_alg = casadi.MX.sym("y_alg", self.concatenated_algebraic.size)
        y_casadi = casadi.vertcat(y_diff, y_alg)

        # Read inputs
        inputs_wrong_order = {}
        for input_param in self.input_parameters:
            name = input_param.name
            inputs_wrong_order[name] = casadi.MX.sym(name, input_param._expected_size)
        # Sort according to input_parameter_order
        if input_parameter_order is None:
            if len(inputs_wrong_order) > 1:
                raise ValueError(
                    "input_parameter_order must be specified if there is more than one "
                    "input parameter"
                )
            inputs = inputs_wrong_order
        else:
            inputs = {name: inputs_wrong_order[name] for name in input_parameter_order}
        # Set up inputs
        inputs_stacked = casadi.vertcat(*[p for p in inputs.values()])

        # Convert initial conditions to casadi form
        y0 = self.concatenated_initial_conditions.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        x0 = y0[: self.concatenated_rhs.size]
        z0 = y0[self.concatenated_rhs.size :]

        # Convert rhs and algebraic to casadi form and calculate jacobians
        rhs = self.concatenated_rhs.to_casadi(t_casadi, y_casadi, inputs=inputs)
        jac_rhs = casadi.jacobian(rhs, y_casadi)
        algebraic = self.concatenated_algebraic.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        jac_algebraic = casadi.jacobian(algebraic, y_casadi)

        # For specified variables, convert to casadi
        variables = OrderedDict()
        for name in variable_names:
            var = self.variables[name]
            variables[name] = var.to_casadi(t_casadi, y_casadi, inputs=inputs)

        casadi_dict = {
            "t": t_casadi,
            "x": y_diff,
            "z": y_alg,
            "inputs": inputs_stacked,
            "rhs": rhs,
            "algebraic": algebraic,
            "jac_rhs": jac_rhs,
            "jac_algebraic": jac_algebraic,
            "variables": variables,
            "x0": x0,
            "z0": z0,
        }

        return casadi_dict

    def generate(
        self, filename, variable_names, input_parameter_order=None, cg_options=None
    ):
        """
        Generate the model in C, using CasADi.

        Parameters
        ----------
        filename : str
            Name of the file to which to save the code
        variable_names : list
            Variables to be exported alongside the model structure
        input_parameter_order : list, optional
            Order in which the input parameters should be stacked.
            If input_parameter_order=None and len(self.input_parameters) > 1, a
            ValueError is raised (this helps to avoid accidentally using the wrong
            order)
        cg_options : dict
            Options to pass to the code generator.
            See https://web.casadi.org/docs/#generating-c-code
        """
        model = self.export_casadi_objects(variable_names, input_parameter_order)

        # Read the exported objects
        t, x, z, p = model["t"], model["x"], model["z"], model["inputs"]
        x0, z0 = model["x0"], model["z0"]
        rhs, alg = model["rhs"], model["algebraic"]
        variables = model["variables"]
        jac_rhs, jac_alg = model["jac_rhs"], model["jac_algebraic"]

        # Create functions
        rhs_fn = casadi.Function("rhs_", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg_", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        # Call these functions to initialize initial conditions
        # (initial conditions are not yet consistent at this stage)
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("z0", [p], [z0])
        # Variables
        variables_stacked = casadi.vertcat(*variables.values())
        variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])

        # Write C files
        cg_options = cg_options or {}
        C = casadi.CodeGenerator(filename, cg_options)
        C.add(rhs_fn)
        C.add(alg_fn)
        C.add(jac_rhs_fn)
        C.add(jac_alg_fn)
        C.add(x0_fn)
        C.add(z0_fn)
        C.add(variables_fn)
        C.generate()

    def latexify(self, filename=None, newline=True, output_variables=None):
        # For docstring, see pybamm.expression_tree.operations.latexify.Latexify
        return Latexify(self, filename, newline).latexify(
            output_variables=output_variables
        )

    # Set :meth:`latexify` docstring from :class:`Latexify`
    latexify.__doc__ = Latexify.__doc__

    def process_parameters_and_discretise(self, symbol, parameter_values, disc):
        """
        Process parameters and discretise a symbol using supplied parameter values
        and discretisation. Note: care should be taken if using spatial operators
        on dimensional symbols. Operators in pybamm are written in non-dimensional
        form, so may need to be scaled by the appropriate length scale. It is
        recommended to use this method on non-dimensional symbols.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values to use during processing
        disc : :class:`pybamm.Discretisation`
            The discrisation to use

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        # Set y slices
        if disc.y_slices == {}:
            variables = list(self.rhs.keys()) + list(self.algebraic.keys())
            for variable in variables:
                variable.bounds = tuple(
                    [
                        parameter_values.process_symbol(bound)
                        for bound in variable.bounds
                    ]
                )
            disc.set_variable_slices(variables)

        # Set boundary conditions (also requires setting parameter values)
        if disc.bcs == {}:
            self.boundary_conditions = parameter_values.process_boundary_conditions(
                self
            )
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol


# helper functions for finding symbols
def find_symbol_in_tree(tree, name):
    if name == tree.name:
        return tree
    elif len(tree.children) > 0:
        for child in tree.children:
            child_return = find_symbol_in_tree(child, name)
            if child_return is not None:
                return child_return


def find_symbol_in_dict(dic, name):
    for tree in dic.values():
        tree_return = find_symbol_in_tree(tree, name)
        if tree_return is not None:
            return tree_return


def find_symbol_in_model(model, name):
    dics = [model.rhs, model.algebraic, model.variables]
    for dic in dics:
        dic_return = find_symbol_in_dict(dic, name)
        if dic_return is not None:
            return dic_return


class EquationDict(dict):
    def __init__(self, name, equations):
        self.name = name
        equations = self.check_and_convert_equations(equations)
        super().__init__(equations)

    def __setitem__(self, key, value):
        """Call the update functionality when doing a setitem."""
        self.update({key: value})

    def update(self, equations):
        equations = self.check_and_convert_equations(equations)
        super().update(equations)

    def check_and_convert_equations(self, equations):
        """
        Convert any scalar equations in dict to 'pybamm.Scalar'
        and check that domains are consistent
        """
        # Convert any numbers to a pybamm.Scalar
        for var, eqn in equations.items():
            if isinstance(eqn, numbers.Number):
                eqn = pybamm.Scalar(eqn)
                equations[var] = eqn
            if not (var.domain == eqn.domain or var.domain == [] or eqn.domain == []):
                raise pybamm.DomainError(
                    "variable and equation in '{}' must have the same domain".format(
                        self.name
                    )
                )

        # For initial conditions, check that the equation doesn't contain any
        # Variable objects
        # skip this if the dictionary has no "name" attribute (which will be the case
        # after pickling)
        if hasattr(self, "name") and self.name == "initial_conditions":
            for var, eqn in equations.items():
                if eqn.has_symbol_of_classes(pybamm.Variable):
                    unpacker = pybamm.SymbolUnpacker(pybamm.Variable)
                    variable_in_equation = list(unpacker.unpack_symbol(eqn))[0]
                    raise TypeError(
                        "Initial conditions cannot contain 'Variable' objects, "
                        "but '{!r}' found in initial conditions for '{}'".format(
                            variable_in_equation, var
                        )
                    )

        return equations


class BoundaryConditionsDict(dict):
    def __init__(self, bcs):
        bcs = self.check_and_convert_bcs(bcs)
        super().__init__(bcs)

    def __setitem__(self, key, value):
        """Call the update functionality when doing a setitem."""
        self.update({key: value})

    def update(self, bcs):
        bcs = self.check_and_convert_bcs(bcs)
        super().update(bcs)

    def check_and_convert_bcs(self, boundary_conditions):
        """Convert any scalar bcs in dict to 'pybamm.Scalar', and check types."""
        # Convert any numbers to a pybamm.Scalar
        for var, bcs in boundary_conditions.items():
            for side, bc in bcs.items():
                if isinstance(bc[0], numbers.Number):
                    # typ is the type of the bc, e.g. "Dirichlet" or "Neumann"
                    eqn, typ = boundary_conditions[var][side]
                    boundary_conditions[var][side] = (pybamm.Scalar(eqn), typ)
                # Check types
                if bc[1] not in ["Dirichlet", "Neumann"]:
                    raise pybamm.ModelError(
                        """
                        boundary condition types must be Dirichlet or Neumann, not '{}'
                        """.format(
                            bc[1]
                        )
                    )

        return boundary_conditions
