import pybamm


class BaseSubModel(pybamm.BaseModel):
    """
    The base class for all submodels. A submodel overrides :meth:`build` to set its
    variables, equations and events, using :meth:`coupled_variable` for variables
    owned by other submodels. Any other methods added to a submodel are private
    (prefixed with '_'), providing a consistent public interface for all submodels.

    Parameters
    ----------
    param: parameter class
        The model parameter symbols
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    name: str
        A string giving the name of the submodel
    external: bool, optional
        Whether the variables defined by the submodel will be provided externally
        by the users. Default is 'False'.
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is None).

    Attributes
    ----------
    param : parameter class
        The model parameter symbols.
    domain : str
        The domain of the submodel, could be either 'Negative', 'Positive', 'Separator', or None.
    name : str
        The name of the submodel.
    external : bool
        A boolean flag indicating whether the variables defined by the submodel will be
        provided externally by the user. Set to False by default.
    options : dict or pybamm.BatteryModelOptions
        A dictionary or an instance of `pybamm.BatteryModelOptions` that stores configuration
        options for the submodel.
    phase_name : str
        A string representing the phase of the submodel, which could be "primary",
        "secondary", or an empty string if there is only one phase.
    phase : str or None
        The current phase of the submodel, which could be "primary", "secondary", or None.
    boundary_conditions : dict
        A dictionary mapping variables to their respective boundary conditions.
    variables : dict
        A dictionary mapping variable names (strings) to expressions or objects that
        represent the useful variables for the submodel.
    """

    def __init__(
        self,
        param,
        domain=None,
        name="Unnamed submodel",
        external=False,
        options=None,
        phase=None,
    ):
        super().__init__(name)
        self.domain = domain
        self.name = name
        self.external = external

        if options is None or type(options) == dict:
            options = pybamm.BatteryModelOptions(options)

        self.options = options

        self.param = param
        if param is None or domain is None:
            self.domain_param = None
        else:
            self.domain_param = param.domain_params[self.domain]
            if phase is not None:
                self.phase_param = self.domain_param.phase_params[phase]

        # Error checks for phase and domain
        self.set_phase(phase)

    def set_phase(self, phase):
        if phase is not None:
            if self.domain is None:
                raise ValueError("Phase must be None if domain is None")
            options_phase = getattr(self.options, self.domain)["particle phases"]
            if options_phase == "1" and phase != "primary":
                raise ValueError("Phase must be 'primary' if there is only one phase")
            elif options_phase == "2" and phase not in ["primary", "secondary"]:
                raise ValueError(
                    "Phase must be either 'primary' or 'secondary' "
                    "if there are two phases"
                )

            if options_phase == "1" and phase == "primary":
                # Only one phase, no need to distinguish between
                # "primary" and "secondary"
                self.phase_name = ""
            else:
                # add a space so that we can use "" or (e.g.) "primary " interchangeably
                # when naming variables
                self.phase_name = phase + " "

        self.phase = phase

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        if domain is not None:
            domain = domain.lower()
        ok_domain_list = ["negative", "separator", "positive", None]
        if domain in ok_domain_list:
            self._domain = domain
            if domain is not None:
                self._Domain = domain.capitalize()
        else:
            raise pybamm.DomainError(
                f"Domain '{domain}' not recognised (must be one of {ok_domain_list})"
            )

    @property
    def domain_Domain(self):
        """Returns a tuple containing the current domain and its capitalized form."""
        return self._domain, self._Domain

    def get_parameter_info(self, by_submodel=False):
        """
        Extracts the parameter information and returns it as a dictionary.
        To get a list of all parameter-like objects without extra information,
        use :py:attr:`model.parameters`.
        """
        raise NotImplementedError(
            "Cannot use get_parameter_info OR print_parameter_info directly on a submodel. "
            "Please use it on the full model."
        )

    _legacy_methods = (
        "get_fundamental_variables",
        "get_coupled_variables",
        "set_rhs",
        "set_algebraic",
        "set_boundary_conditions",
        "set_initial_conditions",
        "add_events_from",
    )

    @property
    def defines_build(self):
        """
        Whether the submodel defines its equations through :meth:`build`.

        Raises
        ------
        :class:`pybamm.ModelError`
            If a class overrides the deprecated methods of a parent that defines
            :meth:`build`, since those overrides would be ignored.
        """
        legacy_class = None
        for cls in type(self).__mro__:
            if cls is BaseSubModel:
                return False
            if "build" in vars(cls):
                if legacy_class is None:
                    return True
                raise pybamm.ModelError(
                    f"{legacy_class.__name__} overrides deprecated submodel methods "
                    f"but inherits `build` from {cls.__name__}, so the overrides "
                    "would be ignored. Define `build` instead."
                )
            if legacy_class is None and any(
                name in vars(cls) for name in self._legacy_methods
            ):
                legacy_class = cls
        return False

    def build(self):
        """
        Set the submodel's variables, equations and events.

        Variables owned by other submodels are referred to with
        :meth:`coupled_variable`, so submodels can be built in any order. Results
        are stored in ``self.variables``, ``self.rhs``, ``self.algebraic``,
        ``self.boundary_conditions``, ``self.initial_conditions`` and
        ``self.events``.
        """
        raise NotImplementedError

    def coupled_variable(
        self, name, domain=None, auxiliary_domains=None, domains=None, on_edges=False
    ):
        """
        Return a placeholder for a variable owned by another submodel.

        The placeholder is replaced by the variable of the same name when the
        model is built, which checks that the domains match.

        Parameters
        ----------
        name : str
            Name of the variable in the model's variables.
        domain : str or list of str, optional
            Primary domain of the variable.
        auxiliary_domains : dict, optional
            Auxiliary domains of the variable.
        domains : dict, optional
            All domains of the variable, as an alternative to ``domain`` and
            ``auxiliary_domains``.
        on_edges : bool, optional
            Whether the variable is evaluated on cell edges, as a flux is.

        Returns
        -------
        :class:`pybamm.CoupledVariable`
        """
        variable = pybamm.CoupledVariable(
            name,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
            domains=domains,
            on_edges=on_edges,
        )
        self.coupled_variables[name] = variable
        return variable

    def get_fundamental_variables(self):
        """
        Deprecated: override :meth:`build` instead.

        A public method that creates and returns the variables in a submodel which can
        be created independent of other submodels. For example, the electrolyte
        concentration variables can be created independent of whether any other
        variables have been defined in the model. As a rule, if a variable can be
        created without variables from other submodels, then it should be placed in
        this method.

        Returns
        -------
        dict :
            The variables created by the submodel which are independent of variables in
            other submodels.
        """
        return {}

    def get_coupled_variables(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A public method that creates and returns the variables in a submodel which
        require variables in other submodels to be set first. For example, the
        exchange current density requires the concentration in the electrolyte to
        be created before it can be created. If a variable can be created independent
        of other submodels then it should be created in 'get_fundamental_variables'
        instead of this method.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.

        Returns
        -------
        dict :
            The variables created in this submodel which depend on variables in
            other submodels.
        """
        return {}

    def set_rhs(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A method to set the right hand side of the differential equations which contain
        a time derivative. Note: this method modifies the state of self.rhs. Unless
        overwritten by a submodel, the default behaviour of 'pass' is used as
        implemented in :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """

    def set_algebraic(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A method to set the differential equations which do not contain a time
        derivative. Note: this method modifies the state of self.algebraic. Unless
        overwritten by a submodel, the default behaviour of 'pass' is used as
        implemented in :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """

    def set_boundary_conditions(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A method to set the boundary conditions for the submodel. Note: this method
        modifies the state of self.boundary_conditions. Unless overwritten by a
        submodel, the default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """

    def set_initial_conditions(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A method to set the initial conditions for the submodel. Note: this method
        modifies the state of self.initial_conditions. Unless overwritten by a
        submodel, the default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.


        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """

    def add_events_from(self, variables):
        """
        Deprecated: override :meth:`build` instead.

        A method to set events related to the state of submodel variable. Note: this
        method modifies the state of self.events. Unless overwritten by a submodel, the
        default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
