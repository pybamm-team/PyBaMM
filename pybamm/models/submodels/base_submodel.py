#
# Base submodel class
#
import pybamm


class BaseSubModel(pybamm.BaseModel):
    """
    The base class for all submodels. All submodels inherit from this class and must
    only provide public methods which overwrite those in this base class. Any methods
    added to a submodel that do not overwrite those in this bass class are made
    private with the prefix '_', providing a consistent public interface for all
    submodels.

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
    param: parameter class
        The model parameter symbols
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
    events: list
        A list of events. Each event can either cause the solver to terminate
        (e.g. concentration goes negative), or be used to inform the solver of the
        existance of a discontinuity (e.g. discontinuity in the input current)
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
                "Domain '{}' not recognised (must be one of {})".format(
                    domain, ok_domain_list
                )
            )

    @property
    def domain_Domain(self):
        return self._domain, self._Domain

    def get_fundamental_variables(self):
        """
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
        A method to set the right hand side of the differential equations which contain
        a time derivative. Note: this method modifies the state of self.rhs. Unless
        overwritten by a submodel, the default behaviour of 'pass' is used as
        implemented in :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass

    def set_algebraic(self, variables):
        """
        A method to set the differential equations which do not contain a time
        derivative. Note: this method modifies the state of self.algebraic. Unless
        overwritten by a submodel, the default behaviour of 'pass' is used as
        implemented in :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass

    def set_boundary_conditions(self, variables):
        """
        A method to set the boundary conditions for the submodel. Note: this method
        modifies the state of self.boundary_conditions. Unless overwritten by a
        submodel, the default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass

    def set_initial_conditions(self, variables):
        """
        A method to set the initial conditions for the submodel. Note: this method
        modifies the state of self.initial_conditions. Unless overwritten by a
        submodel, the default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.


        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass

    def set_events(self, variables):
        """
        A method to set events related to the state of submodel variable. Note: this
        method modifies the state of self.events. Unless overwritten by a submodel, the
        default behaviour of 'pass' is used as implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass
