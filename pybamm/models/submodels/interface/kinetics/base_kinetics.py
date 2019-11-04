#
# Base kinetics class
#

import pybamm
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
    Base submodel for kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        # Calculate delta_phi from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi(variables)
        delta_phi = variables[self.domain + " electrode surface potential difference"]
        # If delta_phi was broadcast, take only the orphan
        if isinstance(delta_phi, pybamm.Broadcast):
            delta_phi = delta_phi.orphans[0]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi - ocp
        # Get number of electrons in reaction
        ne = self._get_number_of_electrons_in_reaction()
        # Get kinetics. Note: T must have the same domain as j0 and eta_r
        if j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
        else:
            T = variables[self.domain + " electrode temperature"]
        j = self._get_kinetics(j0, ne, eta_r, T)
        # Get average interfacial current density
        j_tot_av = self._get_average_total_interfacial_current_density(variables)
        # j = j_tot_av + (j - pybamm.x_average(j))  # enforce true average

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
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

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError

    def _get_kinetics(self, j0, ne, eta_r, T):
        raise NotImplementedError

    def _get_open_circuit_potential(self, variables):
        raise NotImplementedError

    def _get_dj_dc(self, variables):
        """
        Default to calculate derivative of interfacial current density with respect to
        concentration. Can be overwritten by specific kinetic functions.
        """
        c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        j = self._get_kinetics(j0, ne, delta_phi - ocp, T)
        return j.diff(c_e)

    def _get_dj_ddeltaphi(self, variables):
        """
        Default to calculate derivative of interfacial current density with respect to
        surface potential difference. Can be overwritten by specific kinetic functions.
        """
        _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        j = self._get_kinetics(j0, ne, delta_phi - ocp, T)
        return j.diff(delta_phi)

    def _get_interface_variables_for_first_order(self, variables):
        # This is a bit of a hack, but we need to multiply electrolyte concentration by
        # one to differentiate it from the electrolyte concentration inside the
        # surface potential difference when taking j.diff(c_e) later on
        c_e_0 = variables["Leading-order x-averaged electrolyte concentration"] * 1
        hacked_variables = {
            **variables,
            self.domain + " electrolyte concentration": c_e_0,
        }
        delta_phi = variables[
            "Leading-order x-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        j0 = self._get_exchange_current_density(hacked_variables)
        ne = self._get_number_of_electrons_in_reaction()
        ocp = self._get_open_circuit_potential(hacked_variables)[0]
        if j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
        else:
            T = variables[self.domain + " electrode temperature"]
        return c_e_0, delta_phi, j0, ne, ocp, T

    def _get_j_diffusion_limited_first_order(self, variables):
        """
        First-order correction to the interfacial current density due to
        diffusion-limited effects. For a general model the correction term is zero,
        since the reaction is not diffusion-limited
        """
        return pybamm.Scalar(0)
