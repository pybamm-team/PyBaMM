#
# Bulter volmer class
#

import pybamm
from ..base_interface import BaseInterface


class BaseInverseKinetics(BaseInterface):
    """
    A base submodel that implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j.domain is used.

    **Extends:** :class:`pybamm.interface.kinetics.ButlerVolmer`

    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        ocp, dUdT = self._get_open_circuit_potential(variables)

        j0 = self._get_exchange_current_density(variables)
        j_tot_av = self._get_average_total_interfacial_current_density(variables)
        # Broadcast to match j0's domain
        if j0.domain in [[], ["current collector"]]:
            j = j_tot_av
        else:
            j = pybamm.PrimaryBroadcast(j_tot_av, [self.domain.lower() + " electrode"])

        if self.domain == "Negative":
            ne = self.param.ne_n
        elif self.domain == "Positive":
            ne = self.param.ne_p
        # Note: T must have the same domain as j0 and eta_r
        if j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
        else:
            T = variables[self.domain + " electrode temperature"]

        eta_r = self._get_overpotential(j, j0, ne, T)
        delta_phi = eta_r + ocp

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        if (
            "Negative electrode" + self.reaction_name + " interfacial current density"
            in variables
            and "Positive electrode"
            + self.reaction_name
            + " interfacial current density"
            in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables

    def _get_overpotential(self, j, j0, ne, T):
        raise NotImplementedError
