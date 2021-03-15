#
# Base model class
#
import casadi
import numpy as np
import numbers
import pybamm
import warnings
from collections import OrderedDict


class BaseModel(object):
    """Base model class for other models to extend.

    Attributes
    ----------

    name: str
        A string giving the name of the model
    options: dict
        A dictionary of options to be passed to the model
    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
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
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables
    events: list of :class:`pybamm.Event`
        A list of events. Each event can either cause the solver to terminate
        (e.g. concentration goes negative), or be used to inform the solver of the
        existance of a discontinuity (e.g. discontinuity in the input current)
    concatenated_rhs : :class:`pybamm.Concatenation`
        After discretisation, contains the expressions representing the rhs equations
        concatenated into a single expression
    concatenated_algebraic : :class:`pybamm.Concatenation`
        After discretisation, contains the expressions representing the algebraic
        equations concatenated into a single expression
    concatenated_initial_conditions : :class:`numpy.array`
        After discretisation, contains the vector of initial conditions
    mass_matrix : :class:`pybamm.Matrix`
        After discretisation, contains the mass matrix for the model. This is computed
        automatically
    mass_matrix_inv : :class:`pybamm.Matrix`
        After discretisation, contains the inverse mass matrix for the differential
        (rhs) part of model. This is computed automatically
    jacobian : :class:`pybamm.Concatenation`
        Contains the Jacobian for the model. If model.use_jacobian is True, the
        Jacobian is computed automatically during solver set up
    jacobian_rhs : :class:`pybamm.Concatenation`
        Contains the Jacobian for the part of the model which contains time derivatives.
        If model.use_jacobian is True, the Jacobian is computed automatically during
        solver set up
    jacobian_algebraic : :class:`pybamm.Concatenation`
        Contains the Jacobian for the algebraic part of the model. This may be used
        by the solver when calculating consistent initial conditions. If
        model.use_jacobian is True, the Jacobian is computed automatically during
        solver set up
    use_jacobian : bool
        Whether to use the Jacobian when solving the model (default is True)
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
        self.options = {}

        # Initialise empty model
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
        self.external_variables = []
        self._parameters = None
        self._input_parameters = None
        self._variables_casadi = {}

        # Default behaviour is to use the jacobian
        self.use_jacobian = True
        self.convert_to_format = "casadi"

        # Model is not initially discretised
        self.is_discretised = False
        self.y_slices = None

        # Default timescale is 1 second
        self.timescale = pybamm.Scalar(1)
        self.length_scales = {}

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
        self._variables = pybamm.FuzzyDict(variables)

    def variable_names(self):
        return list(self._variables.keys())

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
        """Timescale of model, to be used for non-dimensionalising time when solving"""
        return self._timescale

    @timescale.setter
    def timescale(self, value):
        """Set the timescale"""
        self._timescale = value

    @property
    def parameters(self):
        """Returns all the parameters in the model"""
        if self._parameters is None:
            self._parameters = self._find_parameters()
        return self._parameters

    def _find_parameters(self):
        """Find all the parameters in the model"""
        unpacker = pybamm.SymbolUnpacker((pybamm.Parameter, pybamm.InputParameter))
        all_parameters = unpacker.unpack_list_of_symbols(
            list(self.rhs.values())
            + list(self.algebraic.values())
            + list(self.initial_conditions.values())
            + list(self.variables.values())
            + [event.expression for event in self.events]
        )
        return list(all_parameters.values())

    @property
    def input_parameters(self):
        """Returns all the input parameters in the model"""
        if self._input_parameters is None:
            self._input_parameters = self._find_input_parameters()
        return self._input_parameters

    def _find_input_parameters(self):
        """Find all the input parameters in the model"""
        unpacker = pybamm.SymbolUnpacker(pybamm.InputParameter)
        all_input_parameters = unpacker.unpack_list_of_symbols(
            list(self.rhs.values())
            + list(self.algebraic.values())
            + list(self.initial_conditions.values())
            + list(self.variables.values())
            + [event.expression for event in self.events]
        )
        return list(all_input_parameters.values())

    def __getitem__(self, key):
        return self.rhs[key]

    def new_empty_copy(self):
        """
        Create an empty copy of the model with the same name and "parameters"
        (convert_to_format, etc), but empty equations and variables.
        This is usually then called by :class:`pybamm.ParameterValues`,
        :class:`pybamm.Discretisation`, or :class:`pybamm.SymbolReplacer`.
        """
        new_model = self.__class__(name=self.name)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales

        # Variables from discretisation
        new_model.is_discretised = self.is_discretised
        new_model.y_slices = self.y_slices
        new_model.concatenated_rhs = self.concatenated_rhs
        new_model.concatenated_algebraic = self.concatenated_algebraic
        new_model.concatenated_initial_conditions = self.concatenated_initial_conditions

        return new_model

    def new_copy(self):
        """
        Creates an identical copy of the model, using the functionality of
        :class:`pybamm.SymbolReplacer` but without performing any replacements
        """
        replacer = pybamm.SymbolReplacer({})
        return replacer.process_model(self, inplace=False)

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

    def set_initial_conditions_from(self, solution, inplace=True):
        """
        Update initial conditions with the final states from a Solution object or from
        a dictionary.
        This assumes that, for each variable in self.initial_conditions, there is a
        corresponding variable in the solution with the same name and size.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`, or dict
            The solution to use to initialize the model
        inplace : bool
            Whether to modify the model inplace or create a new model
        """
        if inplace is True:
            model = self
        else:
            model = self.new_copy()

        if isinstance(solution, pybamm.Solution):
            solution = solution.last_state
        for var, equation in model.initial_conditions.items():
            if isinstance(var, pybamm.Variable):
                try:
                    final_state = solution[var.name]
                except KeyError as e:
                    raise pybamm.ModelError(
                        "To update a model from a solution, each variable in "
                        "model.initial_conditions must appear in the solution with "
                        "the same key as the variable name. In the solution provided, "
                        f"{e.args[0]}"
                    )
                if isinstance(solution, pybamm.Solution):
                    final_state = final_state.data
                if final_state.ndim == 1:
                    final_state_eval = final_state[-1:]
                elif final_state.ndim == 2:
                    final_state_eval = final_state[:, -1]
                elif final_state.ndim == 3:
                    final_state_eval = final_state[:, :, -1].flatten(order="F")
                else:
                    raise NotImplementedError("Variable must be 0D, 1D, or 2D")
                model.initial_conditions[var] = pybamm.Vector(final_state_eval)
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
                model.initial_conditions[var] = pybamm.Vector(np.concatenate(children))

            else:
                raise NotImplementedError(
                    "Variable must have type 'Variable' or 'Concatenation'"
                )

        # Also update the concatenated initial conditions if the model is already
        # discretised
        if model.is_discretised:
            # Unpack slices for sorting
            y_slices = {var.id: slce for var, slce in model.y_slices.items()}
            slices = []
            for symbol in model.initial_conditions.keys():
                if isinstance(symbol, pybamm.Concatenation):
                    # must append the slice for the whole concatenation, so that
                    # equations get sorted correctly
                    slices.append(
                        slice(
                            y_slices[symbol.children[0].id][0].start,
                            y_slices[symbol.children[-1].id][0].stop,
                        )
                    )
                else:
                    slices.append(y_slices[symbol.id][0])
            equations = list(model.initial_conditions.values())
            # sort equations according to slices
            sorted_equations = [eq for _, eq in sorted(zip(slices, equations))]
            model.concatenated_initial_conditions = pybamm.NumpyConcatenation(
                *sorted_equations
            )

        return model

    def check_and_combine_dict(self, dict1, dict2):
        # check that the key ids are distinct
        ids1 = set(x.id for x in dict1.keys())
        ids2 = set(x.id for x in dict2.keys())
        if len(ids1.intersection(ids2)) != 0:
            variables = [x for x in dict1.keys() if x.id in ids1.intersection(ids2)]
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
        self.check_default_variables_dictionaries()
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
        vars_in_rhs_keys = set()
        vars_in_algebraic_keys = set()
        vars_in_eqns = set()
        # Get all variables ids from rhs and algebraic keys and equations, and
        # from boundary conditions
        # For equations we look through the whole expression tree.
        # "Variables" can be Concatenations so we also have to look in the whole
        # expression tree
        unpacker = pybamm.SymbolUnpacker((pybamm.Variable, pybamm.VariableDot))

        for var, eqn in self.rhs.items():
            # Find all variables and variabledot objects
            vars_in_rhs_keys_dict = unpacker.unpack_symbol(var)
            vars_in_eqns_dict = unpacker.unpack_symbol(eqn)

            # Store ids only
            # Look only for Variable (not VariableDot) in rhs keys
            vars_in_rhs_keys.update(
                [
                    var_id
                    for var_id, var in vars_in_rhs_keys_dict.items()
                    if isinstance(var, pybamm.Variable)
                ]
            )
            vars_in_eqns.update(vars_in_eqns_dict.keys())
        for var, eqn in self.algebraic.items():
            # Find all variables and variabledot objects
            vars_in_algebraic_keys_dict = unpacker.unpack_symbol(var)
            vars_in_eqns_dict = unpacker.unpack_symbol(eqn)

            # Store ids only
            # Look only for Variable (not VariableDot) in algebraic keys
            vars_in_algebraic_keys.update(
                [
                    var_id
                    for var_id, var in vars_in_algebraic_keys_dict.items()
                    if isinstance(var, pybamm.Variable)
                ]
            )
            vars_in_eqns.update(vars_in_eqns_dict.keys())
        for var, side_eqn in self.boundary_conditions.items():
            for side, (eqn, typ) in side_eqn.items():
                vars_in_eqns_dict = unpacker.unpack_symbol(eqn)
                vars_in_eqns.update(vars_in_eqns_dict.keys())

        # If any keys are repeated between rhs and algebraic then the model is
        # overdetermined
        if not set(vars_in_rhs_keys).isdisjoint(vars_in_algebraic_keys):
            raise pybamm.ModelError("model is overdetermined (repeated keys)")
        # If any algebraic keys don't appear in the eqns (or bcs) then the model is
        # overdetermined (but rhs keys can be absent from the eqns, e.g. dcdt = -1 is
        # fine)
        # Skip this step after discretisation, as any variables in the equations will
        # have been discretised to slices but keys will still be variables
        extra_algebraic_keys = vars_in_algebraic_keys.difference(vars_in_eqns)
        if extra_algebraic_keys and not post_discretisation:
            raise pybamm.ModelError("model is overdetermined (extra algebraic keys)")
        # If any variables in the equations don't appear in the keys then the model is
        # underdetermined
        vars_in_keys = vars_in_rhs_keys.union(vars_in_algebraic_keys)
        extra_variables_in_equations = vars_in_eqns.difference(vars_in_keys)

        # get ids of external variables
        external_ids = {var.id for var in self.external_variables}
        for var in self.external_variables:
            if isinstance(var, pybamm.Concatenation):
                child_ids = {child.id for child in var.children}
                external_ids = external_ids.union(child_ids)

        extra_variables = extra_variables_in_equations.difference(external_ids)

        if extra_variables:
            raise pybamm.ModelError("model is underdetermined (too many variables)")

    def check_algebraic_equations(self, post_discretisation):
        """
        Check that the algebraic equations are well-posed.
        Before discretisation, each algebraic equation key must appear in the equation
        After discretisation, there must be at least one StateVector in each algebraic
        equation
        """
        vars_in_bcs = set()
        unpacker = pybamm.SymbolUnpacker(pybamm.Variable)
        for side_eqn in self.boundary_conditions.values():
            all_vars = unpacker.unpack_list_of_symbols(
                [eqn for eqn, _ in side_eqn.values()]
            )
            vars_in_bcs.update(all_vars.keys())
        if not post_discretisation:
            # After the model has been defined, each algebraic equation key should
            # appear in that algebraic equation, or in the boundary conditions
            # this has been relaxed for concatenations for now
            for var, eqn in self.algebraic.items():
                if not (
                    any(x.id == var.id for x in eqn.pre_order())
                    or var.id in vars_in_bcs
                    or isinstance(var, pybamm.Concatenation)
                ):
                    raise pybamm.ModelError(
                        "each variable in the algebraic eqn keys must appear in the eqn"
                    )
        else:
            # variables in keys don't get discretised so they will no longer match
            # with the state vectors in the algebraic equations. Instead, we check
            # that each algebraic equation contains some StateVector
            for eqn in self.algebraic.values():
                if not eqn.has_symbol_of_classes(pybamm.StateVector):
                    raise pybamm.ModelError(
                        "each algebraic equation must contain at least one StateVector"
                    )

    def check_ics_bcs(self):
        """Check that the initial and boundary conditions are well-posed."""
        # Initial conditions
        for var in self.rhs.keys():
            if var not in self.initial_conditions.keys():
                raise pybamm.ModelError(
                    """no initial condition given for variable '{}'""".format(var)
                )

        # Boundary conditions
        for var, eqn in {**self.rhs, **self.algebraic}.items():
            if eqn.has_symbol_of_classes(
                (pybamm.Gradient, pybamm.Divergence)
            ) and not eqn.has_symbol_of_classes(pybamm.Integral):
                # I have relaxed this check for now so that the lumped temperature
                # equation doesn't raise errors (this has and average in it)

                # Variable must be in the boundary conditions
                if not any(
                    var.id == x.id
                    for symbol in self.boundary_conditions.keys()
                    for x in symbol.pre_order()
                ):
                    raise pybamm.ModelError(
                        "no boundary condition given for "
                        "variable '{}' with equation '{}'.".format(var, eqn)
                    )

    def check_default_variables_dictionaries(self):
        """Check that the right variables are provided."""
        missing_vars = []
        for output, expression in self._variables.items():
            if expression is None:
                missing_vars.append(output)
        if len(missing_vars) > 0:
            warnings.warn(
                "the standard output variable(s) '{}' have not been supplied. "
                "These may be required for testing or comparison with other "
                "models.".format(missing_vars),
                pybamm.ModelWarning,
                stacklevel=2,
            )
            # Remove missing entries
            for output in missing_vars:
                del self._variables[output]

    def check_variables(self):
        # Create list of all Variable nodes that appear in the model's list of variables
        unpacker = pybamm.SymbolUnpacker(pybamm.Variable)
        all_vars = unpacker.unpack_list_of_symbols(self.variables.values())

        var_ids_in_keys = set()

        model_and_external_variables = (
            list(self.rhs.keys())
            + list(self.algebraic.keys())
            + self.external_variables
        )

        for var in model_and_external_variables:
            if isinstance(var, pybamm.Variable):
                var_ids_in_keys.add(var.id)
            # Key can be a concatenation
            elif isinstance(var, pybamm.Concatenation):
                var_ids_in_keys.update([child.id for child in var.children])

        for var_id, var in all_vars.items():
            if var_id not in var_ids_in_keys:
                raise pybamm.ModelError(
                    """
                    No key set for variable '{}'. Make sure it is included in either
                    model.rhs, model.algebraic, or model.external_variables in an
                    unmodified form (e.g. not Broadcasted)
                    """.format(
                        var
                    )
                )

    def check_no_repeated_keys(self):
        """Check that no equation keys are repeated."""
        rhs_alg = {**self.rhs, **self.algebraic}
        rhs_alg_keys = []

        for var in rhs_alg.keys():
            # Check the variable has not already been defined
            if var.id in rhs_alg_keys:
                raise pybamm.ModelError(
                    "Multiple equations specified for variable {!r}".format(var)
                )
            # Update list of variables
            else:
                rhs_alg_keys.append(var.id)

    def info(self, symbol_name):
        """
        Provides helpful summary information for a symbol.

        Parameters
        ----------
        parameter_name : str
        """

        div = "-----------------------------------------"
        symbol = find_symbol_in_model(self, symbol_name)

        if not symbol:
            return None

        print(div)
        print(symbol_name, "\n")
        print(type(symbol))

        if isinstance(symbol, pybamm.FunctionParameter):
            print("")
            print("Inputs:")
            symbol.print_input_names()

        print(div)

    def export_casadi_objects(self, variable_names, input_parameter_order=None):
        """
        Export the constituent parts of the model (rhs, algebraic, initial conditions,
        etc) as casadi objects.

        Parameters
        ----------
        variable_names : list
            Variables to be exported alongside the model structure
        input_parameter_order : list, optional
            Order in which the input parameters should be stacked. If None, the order
            returned by :meth:`BaseModel.input_parameters` is used

        Returns
        -------
        casadi_dict : dict
            Dictionary of {str: casadi object} pairs representing the model in casadi
            format
        """
        # Discretise model if it isn't already discretised
        # This only works with purely 0D models, as otherwise the mesh and spatial
        # method should be specified by the user
        if self.is_discretised is False:
            try:
                disc = pybamm.Discretisation()
                disc.process_model(self)
            except pybamm.DiscretisationError as e:
                raise pybamm.DiscretisationError(
                    "Cannot automatically discretise model, model should be "
                    "discretised before exporting casadi functions ({})".format(e)
                )

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
        # Read external variables
        external_casadi = {}
        for external_varaiable in self.external_variables:
            name = external_varaiable.name
            ev_size = external_varaiable._evaluate_for_shape().shape[0]
            external_casadi[name] = casadi.MX.sym(name, ev_size)
        # Sort according to input_parameter_order
        if input_parameter_order is None:
            inputs = inputs_wrong_order
        else:
            inputs = {name: inputs_wrong_order[name] for name in input_parameter_order}
        # Set up external variables and inputs
        # Put external variables first like the integrator expects
        ext_and_in = {**external_casadi, **inputs}
        inputs_stacked = casadi.vertcat(*[p for p in ext_and_in.values()])

        # Convert initial conditions to casadi form
        y0 = self.concatenated_initial_conditions.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        x0 = y0[: self.concatenated_rhs.size]
        z0 = y0[self.concatenated_rhs.size :]

        # Convert rhs and algebraic to casadi form and calculate jacobians
        rhs = self.concatenated_rhs.to_casadi(t_casadi, y_casadi, inputs=ext_and_in)
        jac_rhs = casadi.jacobian(rhs, y_casadi)
        algebraic = self.concatenated_algebraic.to_casadi(
            t_casadi, y_casadi, inputs=inputs
        )
        jac_algebraic = casadi.jacobian(algebraic, y_casadi)

        # For specified variables, convert to casadi
        variables = OrderedDict()
        for name in variable_names:
            var = self.variables[name]
            variables[name] = var.to_casadi(t_casadi, y_casadi, inputs=ext_and_in)

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
            Order in which the input parameters should be stacked. If None, the order
            returned by :meth:`BaseModel.input_parameters` is used
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

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({})

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
        """Return default solver based on whether model is ODE model or DAE model."""
        return pybamm.CasadiSolver(mode="safe")


# helper functions for finding symbols
def find_symbol_in_tree(tree, name):
    if name == tree.name:
        return tree
    elif len(tree.children) > 0:
        for child in tree.children:
            child_return = find_symbol_in_tree(child, name)
            if child_return:
                return child_return


def find_symbol_in_dict(dic, name):
    for tree in dic.values():
        tree_return = find_symbol_in_tree(tree, name)
        if tree_return:
            return tree_return


def find_symbol_in_model(model, name):
    dics = [model.rhs, model.algebraic, model.variables]
    for dic in dics:
        dic_return = find_symbol_in_dict(dic, name)
        if dic_return:
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
                equations[var] = pybamm.Scalar(eqn)

        if not all(
            [
                variable.domain == equation.domain
                or variable.domain == []
                or equation.domain == []
                for variable, equation in equations.items()
            ]
        ):
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
                    variable_in_equation = list(unpacker.unpack_symbol(eqn).values())[0]
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
