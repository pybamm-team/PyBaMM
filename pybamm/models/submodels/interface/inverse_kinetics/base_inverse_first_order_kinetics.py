#
# First-order Butler-Volmer kinetics
#

from ..kinetics.base_first_order_kinetics import BaseFirstOrderKinetics


class BaseInverseFirstOrderKinetics(BaseFirstOrderKinetics):
    """
    Base inverse first-order kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.kinetics.BaseFirstOrderKinetics`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_die1dx(self, variables):
        i_boundary_cc = variables["Current collector current density"]
        i_boundary_cc_0 = variables["Leading-order current collector current density"]
        i_boundary_cc_1 = (i_boundary_cc - i_boundary_cc_0) / self.param.C_e

        if self.domain == "Negative":
            return i_boundary_cc_1 / self.param.l_n
        elif self.domain == "Positive":
            return -i_boundary_cc_1 / self.param.l_p

    def get_coupled_variables(self, variables):
        # Unpack
        delta_phi_0 = variables[
            "Leading-order x-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        c_e_0 = variables["Leading-order x-averaged electrolyte concentration"]
        c_e_av = variables[
            "X-averaged " + self.domain.lower() + " electrolyte concentration"
        ]
        c_e_1_av = (c_e_av - c_e_0) / self.param.C_e

        # Get first-order current (this is zero in 1D)
        die1_dx = self._get_die1dx(variables)

        # Get derivatives of leading-order terms
        sum_dj_dc_0 = sum(
            reaction_submodel._get_dj_dc(variables)
            for reaction_submodel in self.reaction_submodels
        )
        sum_dj_ddeltaphi_0 = sum(
            reaction_submodel._get_dj_ddeltaphi(variables)
            for reaction_submodel in self.reaction_submodels
        )
        sum_j_diffusion_limited_first_order = sum(
            reaction_submodel._get_j_diffusion_limited_first_order(variables)
            for reaction_submodel in self.reaction_submodels
        )

        delta_phi_1_av = (
            die1_dx - (sum_dj_dc_0 * c_e_1_av + sum_j_diffusion_limited_first_order)
        ) / sum_dj_ddeltaphi_0
        delta_phi = delta_phi_0 + self.param.C_e * delta_phi_1_av

        # Update variables dictionary
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )

        return variables
