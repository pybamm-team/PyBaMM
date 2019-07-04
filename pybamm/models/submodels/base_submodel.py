#
# Base submodel class
#
import pybamm


class BaseSubModel:
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
    events: dict
        A dictionary of events that should cause the solver to terminate (e.g.
        concentration goes negative). The keys are strings and the values are
        symbols.
    """

    def __init__(self, param):
        super().__init__()
        self.param = param
        # Initialise empty variables (to avoid overwriting with 'None')

        self.rhs = {}
        self.algebraic = {}
        self.boundary_conditions = {}
        self.initial_conditions = {}
        self.variables = {}
        self.events = {}

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
        submodel, the default behaviour of 'pass' is used a implemented in
        :class:`pybamm.BaseSubModel`.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        pass

    def set_internal_boundary_conditions(self):
        """
        A method to set the internal boundary conditions for the submodel.
        These are required to properly calculate the gradient.
        Note: this method modifies the state of self.boundary_conditions.
        """

        def boundary_average(left_orphan, right_orphan):
            left = pybamm.Index(left_orphan, -1)
            right = pybamm.Index(right_orphan, 0)
            x_n = pybamm.standard_spatial_vars.x_n_edge
            x_s = pybamm.standard_spatial_vars.x_s_edge

            dx_n = pybamm.Index(x_n, -1) - pybamm.Index(x_n, -2)
            dx_s = pybamm.Index(x_s, 1) - pybamm.Index(x_s, 0)

            av = (left * dx_n + right * dx_s) / (dx_n + dx_s)

            x_n_end = pybamm.Index(pybamm.standard_spatial_vars.x_n, -1)
            x_s_0 = pybamm.Index(pybamm.standard_spatial_vars.x_s, 0)

            dy = right - left
            dx = x_s_0 - x_n_end

            return dy / dx

        internal_bcs = {}
        for var in self.boundary_conditions.keys():
            if isinstance(var, pybamm.Concatenation):
                children = var.children
                first_child = children[0]
                middle_children = children[1:-1]
                last_child = children[-1]

                first_orphan = first_child.new_copy()
                lbc = self.boundary_conditions[var]["left"]

                second_orphan = middle_children[0].new_copy()
                rbc = (boundary_average(first_orphan, second_orphan), "Neumann")
                # rbc = (pybamm.BoundaryValue(first_orphan, "right"), "Dirichlet")
                internal_bcs.update({first_child: {"left": lbc, "right": rbc}})

                for i, _ in enumerate(middle_children):

                    previous_orphan = children[i].new_copy()

                    current_child = children[i + 1]
                    current_orphan = current_child.new_copy()

                    next_orphan = children[i + 2].new_copy()

                    lbc = (boundary_average(previous_orphan, current_orphan), "Neumann")
                    rbc = (boundary_average(current_orphan, next_orphan), "Neumann")
                    # lbc = (pybamm.BoundaryValue(current_orphan, "left"), "Dirichlet")
                    # rbc = (pybamm.BoundaryValue(current_orphan, "right"), "Dirichlet")

                    internal_bcs.update({current_child: {"left": lbc, "right": rbc}})

                second_last_orphan = children[-2].new_copy()
                last_orphan = last_child.new_copy()

                # lbc = (pybamm.BoundaryValue(last_orphan, "left"), "Dirichlet")
                lbc = (boundary_average(second_last_orphan, last_orphan), "Neumann")
                rbc = self.boundary_conditions[var]["right"]

                internal_bcs.update({last_child: {"left": lbc, "right": rbc}})

        self.boundary_conditions.update(internal_bcs)

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

