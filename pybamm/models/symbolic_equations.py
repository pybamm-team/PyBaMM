#
# Base class containing the symbolic equations for a model
#
import numbers
import pybamm


class _SymbolicEquations(pybamm._BaseEquations):
    """
    Base class containing the symbolic equations for a model

    **Extends:** :class:`pybamm._BaseEquations`
    """

    def __init__(self):
        self._built = False
        self._built_fundamental_and_external = False

        # Initialise empty model
        super().__init__(
            rhs={},
            algebraic={},
            initial_conditions={},
            boundary_conditions={},
            variables=pybamm.FuzzyDict({}),
            events=[],
            external_variables=[],
            # Default timescale is 1 second
            timescale=pybamm.Scalar(1),
            length_scales={},
        )

    @pybamm._BaseEquations.rhs.setter
    def rhs(self, rhs):
        self._rhs = _EquationDict("rhs", rhs)

    @pybamm._BaseEquations.algebraic.setter
    def algebraic(self, algebraic):
        self._algebraic = _EquationDict("algebraic", algebraic)

    @pybamm._BaseEquations.initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        self._initial_conditions = _EquationDict(
            "initial_conditions", initial_conditions
        )

    @pybamm._BaseEquations.boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        self._boundary_conditions = _BoundaryConditionsDict(boundary_conditions)

    @pybamm._BaseEquations.variables.setter
    def variables(self, variables):
        for name, var in variables.items():
            if (
                isinstance(var, pybamm.Variable)
                and var.name != name
                # Exception if the variable is also there under its own name
                and not (var.name in variables and variables[var.name] == var)
                # Exception for the key "Leading-order"
                and "leading-order" not in var.name.lower()
                and "leading-order" not in name.lower()
            ):
                raise ValueError(
                    f"Variable with name '{var.name}' is in variables dictionary with "
                    f"name '{name}'. Names must match."
                )
        self._variables = pybamm.FuzzyDict(variables)

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

    @pybamm._BaseEquations.events.setter
    def events(self, events):
        self._events = events

    @pybamm._BaseEquations.external_variables.setter
    def external_variables(self, external_variables):
        self._external_variables = external_variables

    @pybamm._BaseEquations.timescale.setter
    def timescale(self, value):
        """Set the timescale"""
        self._timescale = value

    @pybamm._BaseEquations.length_scales.setter
    def length_scales(self, values):
        "Set the length scale, converting any numbers to pybamm.Scalar"
        for domain, scale in values.items():
            if isinstance(scale, numbers.Number):
                values[domain] = pybamm.Scalar(scale)
        self._length_scales = values

    def build_fundamental_and_external(self, model):
        # Get the fundamental variables
        for submodel_name, submodel in model.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, model.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        # Set the submodels that are external
        for sub in model.options["external submodels"]:
            submodels[sub].external = True

        # Set any external variables
        self.external_variables = []
        for submodel_name, submodel in model.submodels.items():
            pybamm.logger.debug(
                "Getting external variables for {} submodel ({})".format(
                    submodel_name, model.name
                )
            )
            external_variables = submodel.get_external_variables()

            self.external_variables += external_variables

        self._built_fundamental_and_external = True

    def build_coupled_variables(self, model):
        # Note: pybamm will try to get the coupled variables for the submodels in the
        # order they are set by the user. If this fails for a particular submodel,
        # return to it later and try again. If setting coupled variables fails and
        # there are no more submodels to try, raise an error.
        submodels = list(model.submodels.keys())
        count = 0
        # For this part the FuzzyDict of variables is briefly converted back into a
        # normal dictionary for speed with KeyErrors
        self._variables = dict(self._variables)
        while len(submodels) > 0:
            count += 1
            for submodel_name, submodel in model.submodels.items():
                if submodel_name in submodels:
                    pybamm.logger.debug(
                        "Getting coupled variables for {} submodel ({})".format(
                            submodel_name, model.name
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

    def build_model_equations(self, model):
        # Set model equations
        for submodel_name, submodel in model.submodels.items():
            if submodel.external is False:
                pybamm.logger.verbose(
                    "Setting rhs for {} submodel ({})".format(submodel_name, model.name)
                )

                submodel.set_rhs(self.variables)
                pybamm.logger.verbose(
                    "Setting algebraic for {} submodel ({})".format(
                        submodel_name, model.name
                    )
                )

                submodel.set_algebraic(self.variables)
                pybamm.logger.verbose(
                    "Setting boundary conditions for {} submodel ({})".format(
                        submodel_name, model.name
                    )
                )

                submodel.set_boundary_conditions(self.variables)
                pybamm.logger.verbose(
                    "Setting initial conditions for {} submodel ({})".format(
                        submodel_name, model.name
                    )
                )
                submodel.set_initial_conditions(self.variables)
                submodel.set_events(self.variables)
                pybamm.logger.verbose(
                    "Updating {} submodel ({})".format(submodel_name, model.name)
                )
                self.update(submodel)
                self.check_no_repeated_keys()

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
            self._variables.update(submodel.variables)  # keys are strings so no check
            self._events += submodel.events

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


class _EquationDict(dict):
    def __init__(self, name, equations):
        name = name
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
                        name
                    )
                )

        # For initial conditions, check that the equation doesn't contain any
        # Variable objects
        # skip this if the dictionary has no "name" attribute (which will be the case
        # after pickling)
        if hasattr(self, "name") and name == "initial_conditions":
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


class _BoundaryConditionsDict(dict):
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
