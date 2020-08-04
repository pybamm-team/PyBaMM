#
# Base submodel class
#
import pybamm
from typing import Dict


class BaseCoupledSubModel(pybamm.BaseSubModel):
    """Base class for coupled submodels.

    Couples submodels that require a more intimate modification of their equations,
    boundary or initial conditions than that allowed via coupled variables. A classic
    example is the case of having two processes that can be considered in isolation but
    that result in more complex physics if they are considered together. Rather than
    modifying the underlying submodels, the coupling can be added via this submodel,
    overriding the parts of the equations or boundary conditions that are required in
    order to implement the coupling.

    Eg. If var_x and var_y are variables of submodels X and Y respectively, then a
    possible coupling affecting the RHS could be:

    rhs[var_x] = rhs[var_x] + f(var_x, var_y)
    rhs[var_y] = rhs[var_y] + g(var_x, var_y)

    If nothing is overriden, the result of using this BaseCoupledSubModel with two or
    more submodels should be identical to adding those submodels independently. Also,
    care must be taken if one of the submodels depends on variables defined in the
    others. In that case, the order in which the submodels are added becomes essential.

    Parameters
    ----------
    submodels: dict
        Dictionary of instances of submodels

    Attributes
    ----------
    submodels: dict
        Dictionary of instances of submodels
    """

    def __init__(
        self,
        param,
        submodels: Dict[str, pybamm.BaseSubModel],
        domain=None,
        name="Unnamed submodel",
        external=False,
    ):
        super().__init__(param, domain, name, external)
        self.submodels: Dict[str, pybamm.BaseSubModel] = submodels

    def get_fundamental_variables(self) -> Dict:
        """Provides the fundamental variables of the coupled submodels, at least.

        If, additionally, other fundamental variables need to be provided, override this
        method, adding those extra variables together with those required by the coupled
        submodels.

        Returns
        -------
        dict :
            The variables created by the submodels which are independent of variables in
            other submodels.
        """
        return self._get_fundamental_variables()

    def get_coupled_variables(self, variables: Dict) -> Dict:
        """Provides the coupled variables of the coupled submodels, at least.

        If, additionally, other coupled variables need to be provided, override this
        method, adding those extra variables together with those required by the coupled
        submodels.

        Returns
        -------
        dict :
            The variables created by the submodels which are dependent of variables in
            other submodels.
        """
        return self._get_coupled_variables(variables)

    def set_rhs(self, variables: Dict) -> None:
        """Sets the right hand side of the differential equations.

        By default, this method will gather the rhs of the coupled submodels, but
        chances are that the coupling between submodels requires to modify those. If
        so, modify the required entry in self.rhs dictionary accordingly. Also, add
        more entries for other equations, if needed.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        self._set_rhs(variables)

    def set_algebraic(self, variables: Dict) -> None:
        """Sets the algebraic equations.

        By default, this method will gather the algebraic equations of the coupled
        submodels, but chances are that the coupling between submodels requires to
        modify those. If so, modify the required entry in self.algebraic dictionary
        accordingly. Also, add more entries for other equations, if needed.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        self._set_algebraic(variables)

    def set_boundary_conditions(self, variables: Dict) -> None:
        """Sets the boundary conditions.

        By default, this method will gather the boundary conditions of the coupled
        submodels, but chances are that the coupling between submodels requires to
        modify those. If so, modify the required entry in self.boundary_conditions
        dictionary accordingly. Also, add more entries for other conditions, if needed.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        self._set_boundary_conditions(variables)

    def set_initial_conditions(self, variables: Dict) -> None:
        """Sets the initial conditions.

        By default, this method will gather the initial conditions of the coupled
        submodels, but chances are that the coupling between submodels requires to
        modify those. If so, modify the required entry in self.initial_conditions
        dictionary accordingly. Also, add more entries for other conditions, if needed.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        self._set_initial_conditions(variables)

    def set_events(self, variables: Dict) -> None:
        """A method to set events related to the state of submodel variable.

        By default, this method will gather the initial conditions of the coupled
        submodels. It is not likely you will want to modify them, but chances are that
        new events need to be added.

        Parameters
        ----------
        variables: dict
            The variables in the whole model.
        """
        self._set_events(variables)

    def _get_fundamental_variables(self) -> Dict:
        variables = {}
        for s in self.submodels.values():
            var = s.get_fundamental_variables()
            if isinstance(var, dict):
                variables.update(var)
        return variables

    def _get_coupled_variables(self, variables: Dict) -> Dict:
        for s in self.submodels.values():
            variables = s.get_coupled_variables(variables)
        return variables

    def _set_rhs(self, variables: Dict) -> None:
        for s in self.submodels.values():
            s.set_rhs(variables)
            self.rhs.update(s.rhs)

    def _set_algebraic(self, variables: Dict) -> None:
        for s in self.submodels.values():
            s.set_algebraic(variables)
            self.algebraic.update(s.algebraic)

    def _set_boundary_conditions(self, variables: Dict) -> None:
        for s in self.submodels.values():
            s.set_boundary_conditions(variables)
            self.boundary_conditions.update(s.boundary_conditions)

    def _set_initial_conditions(self, variables: Dict) -> None:
        for s in self.submodels.values():
            s.set_initial_conditions(variables)
            self.initial_conditions.update(s.initial_conditions)

    def _set_events(self, variables: Dict) -> None:
        for s in self.submodels.values():
            s.set_events(variables)
            self.events.update(s.events)
