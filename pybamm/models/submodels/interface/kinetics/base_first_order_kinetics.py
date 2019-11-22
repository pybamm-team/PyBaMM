#
# First-order Butler-Volmer kinetics
#
from .base_kinetics import BaseModel


class BaseFirstOrderKinetics(BaseModel):
    """
    Base first-order kinetics

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
        # Unpack
        c_e_0 = variables[
            "Leading-order " + self.domain.lower() + " electrolyte concentration"
        ]
        c_e = variables[self.domain + " electrolyte concentration"]
        c_e_1 = (c_e - c_e_0) / self.param.C_e

        dj_dc_0 = self._get_dj_dc(variables)
        dj_ddeltaphi_0 = self._get_dj_ddeltaphi(variables)

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
