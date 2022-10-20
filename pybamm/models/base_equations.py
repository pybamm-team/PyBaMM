#
# Base equations
#
import copy
import pybamm
import functools


class _BaseEquations:
    """
    Base class containing equations defining a model.

    Attributes
    ----------
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
    """

    def __init__(
        self,
        rhs,
        algebraic,
        initial_conditions,
        boundary_conditions,
        variables,
        events,
        external_variables,
        timescale,
        length_scales,
    ):
        # Initialise model with read-only attributes
        self._rhs = rhs
        self._algebraic = algebraic
        self._initial_conditions = initial_conditions
        self._boundary_conditions = boundary_conditions
        self._variables = variables
        self._events = events
        self._external_variables = external_variables
        self._timescale = timescale
        self._length_scales = length_scales

        self._input_parameters = None

    @property
    def rhs(self):
        return self._rhs

    @property
    def algebraic(self):
        return self._algebraic

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @property
    def variables(self):
        return self._variables

    def variable_names(self):
        return list(self._variables.keys())

    @functools.cached_property
    def variables_and_events(self):
        """
        Returns variables and events in a single dictionary
        """
        variables_and_events = self.variables.copy()
        variables_and_events.update(
            {f"Event: {event.name}": event.expression for event in self.events}
        )
        return variables_and_events

    @property
    def events(self):
        return self._events

    @property
    def external_variables(self):
        return self._external_variables

    @property
    def timescale(self):
        """Timescale of model, to be used for non-dimensionalising time when solving"""
        return self._timescale

    @property
    def length_scales(self):
        "Length scales of model"
        return self._length_scales

    def copy(self):
        """
        Creates a copy of the model, explicitly copying all the mutable attributes
        to avoid issues with shared objects.
        """
        new_equations = copy.copy(self)
        new_equations._rhs = self.rhs.copy()
        new_equations._algebraic = self.algebraic.copy()
        new_equations._initial_conditions = self.initial_conditions.copy()
        new_equations._boundary_conditions = self.boundary_conditions.copy()
        new_equations._variables = self.variables.copy()
        new_equations._events = self.events.copy()
        new_equations._external_variables = self.external_variables.copy()
        return new_equations

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

        # get external variables
        external_vars = set(self.external_variables)
        for var in self.external_variables:
            if isinstance(var, pybamm.Concatenation):
                child_vars = set(var.children)
                external_vars = external_vars.union(child_vars)

        extra_variables = extra_variables_in_equations.difference(external_vars)

        if extra_variables:
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

    def check_default_variables_dictionaries(self):
        """Check that the right variables are provided."""
        missing_vars = []
        for output, expression in self.variables.items():
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

        vars_in_keys = set()

        model_and_external_variables = (
            list(self.rhs.keys())
            + list(self.algebraic.keys())
            + self.external_variables
        )

        for var in model_and_external_variables:
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
                    model.rhs, model.algebraic, or model.external_variables in an
                    unmodified form (e.g. not Broadcasted)
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

    @functools.cached_property
    def parameters(self):
        """Returns all the parameters in the model"""
        return self._find_symbols(
            (pybamm.Parameter, pybamm.InputParameter, pybamm.FunctionParameter)
        )

    @functools.cached_property
    def input_parameters(self):
        """Returns all the input parameters in the model"""
        return self._find_symbols(pybamm.InputParameter)

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
            + [self.timescale]
            + list(self.length_scales.values())
        )
        return list(all_input_parameters)


class _BaseProcessedEquations(_BaseEquations):
    """
    Base class containing equations defining a model that has been created by processing
    another model. Attributes are read-only.
    Variables are parameterized "on the fly" when they are called.

    **Extends:** :class:`pybamm._BaseEquations`
    """

    def __init__(
        self,
        rhs,
        algebraic,
        initial_conditions,
        boundary_conditions,
        unprocessed_variables,
        events,
        external_variables,
        timescale,
        length_scales,
    ):
        super().__init__(
            # Initialise model with read-only attributes
            rhs=pybamm.ReadOnlyDict(rhs),
            algebraic=pybamm.ReadOnlyDict(algebraic),
            initial_conditions=pybamm.ReadOnlyDict(initial_conditions),
            boundary_conditions=pybamm.ReadOnlyDict(boundary_conditions),
            # Variables is initially empty, but will be filled in when variables are
            # called
            variables=_OnTheFlyUpdatedDict(
                unprocessed_variables, self.variables_update_function
            ),
            events=events,
            external_variables=external_variables,
            timescale=timescale,
            length_scales=length_scales,
        )

    @_BaseEquations.rhs.setter
    def rhs(self, value):
        raise AttributeError(f"Attributes of {self} are read-only")


class _OnTheFlyUpdatedDict(dict):
    """
    A dictionary that updates itself when a key is called.
    """

    def __init__(self, unprocessed_variables, variables_update_function):
        super().__init__({})
        self.unprocessed_variables = unprocessed_variables
        self.variables_update_function = variables_update_function

    def __getitem__(self, key):
        if key not in self:
            self.update(
                {key: self.variables_update_function(self.unprocessed_variables[key])}
            )
        return super().__getitem__(key)

    def copy(self):
        return self.__class__(
            self.unprocessed_variables, self.variables_update_function
        )
