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
        self, param, domain=None, name="Unnamed submodel", external=False, options=None
    ):
        super().__init__(name)
        self.param = param

        self.domain = domain
        self.set_domain_for_broadcast()
        self.name = name

        self.external = external
        self.options = options or {}

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        ok_domain_list = [
            "Negative",
            "Separator",
            "Positive",
            "Negative electrode",
            "Negative electrolyte",
            "Separator electrolyte",
            "Positive electrode",
            "Positive electrolyte",
        ]
        if domain in ok_domain_list:
            self._domain = domain
        elif domain is None:
            pass
        else:
            raise pybamm.DomainError(
                "Domain '{}' not recognised (must be one of {})".format(
                    domain, ok_domain_list
                )
            )

    def set_domain_for_broadcast(self):
        if hasattr(self, "_domain"):
            if self.domain in ["Negative", "Positive"]:
                self.domain_for_broadcast = self.domain.lower() + " electrode"
            elif self.domain == "Separator":
                self.domain_for_broadcast = "separator"

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

    def get_external_variables(self):
        """
        A public method that returns the variables in a submodel which are
        supplied by an external source.

        Returns
        -------
        list :
            A list of the external variables in the model.
        """

        external_variables = []
        list_of_vars = []

        if self.external is True:
            # look through all the variables in the submodel and get the
            # variables which are state vectors
            submodel_variables = self.get_fundamental_variables()
            for var in submodel_variables.values():
                if isinstance(var, pybamm.Variable):
                    list_of_vars += [var]

                elif isinstance(var, pybamm.Concatenation):
                    if all(
                        isinstance(child, pybamm.Variable) for child in var.children
                    ):
                        list_of_vars += [var]

            # first add only unique concatenations
            unique_ids = []
            for var in list_of_vars:
                if var.id not in unique_ids and isinstance(var, pybamm.Concatenation):
                    external_variables += [var]
                    unique_ids += [var.id]
                    # also add the ids of the children to unique ids
                    for child in var.children:
                        unique_ids += [child.id]

            # now add any unique variables that are not part of a concatentation
            for var in list_of_vars:
                if var.id not in unique_ids:
                    external_variables += [var]
                    unique_ids += [var.id]

        return external_variables

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
        submodel, the default behaviour of 'pass' is used a implemented in
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
        submodel, the default behaviour of 'pass' is used a implemented in
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
        default behaviour of 'pass' is used a implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass
