#
# First-order Butler-Volmer kinetics
#
import pybamm
from ..base_interface import BaseInterface


class FirstOrderKinetics(BaseInterface):
    """
    First-order kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    leading_order_model : :class:`pybamm.interface.kinetics.BaseKinetics`
        The leading-order model with respect to which this is first-order

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain, leading_order_model):
        super().__init__(param, domain, leading_order_model.reaction)
        self.leading_order_model = leading_order_model

    def get_coupled_variables(self, variables):
        # Unpack
        c_e_0 = variables[
            "Leading-order " + self.domain.lower() + " electrolyte concentration"
        ]
        c_e = variables[self.domain + " electrolyte concentration"]
        c_e_1 = (c_e - c_e_0) / self.param.C_e

        dj_dc_0 = self.leading_order_model._get_dj_dc(variables)
        dj_ddeltaphi_0 = self.leading_order_model._get_dj_ddeltaphi(variables)

        # Update delta_phi with new phi_e and phi_s
        variables = self._get_delta_phi(variables)

        delta_phi_0 = variables[
            "Leading-order "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        delta_phi = variables[self.domain + " electrode surface potential difference"]
        delta_phi_1 = (delta_phi - delta_phi_0) / self.param.C_e

        j_0 = variables[
            "Leading-order "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
        ]
        j_1 = dj_dc_0 * c_e_1 + dj_ddeltaphi_0 * delta_phi_1
        j = j_0 + self.param.C_e * j_1
        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi - ocp

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        # SEI film resistance not implemented in this model
        eta_sei = pybamm.Scalar(0)
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))

        # Add first-order averages
        j_1_bar = dj_dc_0 * pybamm.x_average(c_e_1) + dj_ddeltaphi_0 * pybamm.x_average(
            delta_phi_1
        )

        variables.update(
            {
                "First-order x-averaged "
                + self.domain.lower()
                + " electrode"
                + self.reaction_name
                + " interfacial current density": j_1_bar
            }
        )

        if self.domain == "Positive":
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables
