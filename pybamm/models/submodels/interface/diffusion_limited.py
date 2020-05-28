#
# Leading-order diffusion limited kinetics
#

import pybamm
from .base_interface import BaseInterface


class DiffusionLimited(BaseInterface):
    """
    Submodel for diffusion-limited kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    order : str
        The order of the model ("leading" or "full")

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain, reaction, order):
        super().__init__(param, domain, reaction)
        self.order = order

    def get_coupled_variables(self, variables):
        # Calculate delta_phi_s from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi(variables)
        delta_phi_s = variables[self.domain + " electrode surface potential difference"]
        # If delta_phi_s was broadcast, take only the orphan
        if isinstance(delta_phi_s, pybamm.Broadcast):
            delta_phi_s = delta_phi_s.orphans[0]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi_s - ocp

        # Get interfacial current densities
        j = self._get_diffusion_limited_current_density(variables)
        j_tot_av = self._get_average_total_interfacial_current_density(variables)

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        # No SEI film resistance in this model
        eta_sei = pybamm.Scalar(0)
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))

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

        if self.order == "composite":
            # For the composite model, adds the first-order x-averaged interfacial
            # current density to the dictionary of variables.
            j_0 = variables[
                "Leading-order "
                + self.domain.lower()
                + " electrode"
                + self.reaction_name
                + " interfacial current density"
            ]
            j_1_bar = (pybamm.x_average(j) - pybamm.x_average(j_0)) / self.param.C_e

            variables.update(
                {
                    "First-order x-averaged "
                    + self.domain.lower()
                    + " electrode"
                    + self.reaction_name
                    + " interfacial current density": j_1_bar
                }
            )

        return variables

    def _get_diffusion_limited_current_density(self, variables):
        param = self.param
        if self.domain == "Negative":
            if self.order == "leading":
                j_p = variables[
                    "X-averaged positive electrode"
                    + self.reaction_name
                    + " interfacial current density"
                ]
                j = -self.param.l_p * j_p / self.param.l_n
            elif self.order in ["composite", "full"]:
                tor_s = variables["Separator tortuosity"]
                c_ox_s = variables["Separator oxygen concentration"]
                N_ox_neg_sep_interface = (
                    -pybamm.boundary_value(tor_s, "left")
                    * param.curlyD_ox
                    * pybamm.BoundaryGradient(c_ox_s, "left")
                )
                N_ox_neg_sep_interface.domain = ["current collector"]

                j = -N_ox_neg_sep_interface / param.C_e / -param.s_ox_Ox / param.l_n

        return j

    def _get_dj_dc(self, variables):
        return pybamm.Scalar(0)

    def _get_dj_ddeltaphi(self, variables):
        return pybamm.Scalar(0)

    def _get_j_diffusion_limited_first_order(self, variables):
        """
        First-order correction to the interfacial current density due to
        diffusion-limited effects. For a general model the correction term is zero,
        since the reaction is not diffusion-limited
        """
        if self.order == "leading":
            j_leading_order = variables[
                "Leading-order x-averaged "
                + self.domain.lower()
                + " electrode"
                + self.reaction_name
                + " interfacial current density"
            ]
            param = self.param
            if self.domain == "Negative":
                N_ox_s_p = variables["Oxygen flux"].orphans[1]
                N_ox_neg_sep_interface = N_ox_s_p[0]

                j = -N_ox_neg_sep_interface / param.C_e / -param.s_ox_Ox / param.l_n

            return (j - j_leading_order) / param.C_e
        else:
            return pybamm.Scalar(0)
