#
# Base class for higher order electrolyte potential employing stefan-maxwell
#
import pybamm
from .base_stefan_maxwell_conductivity import BaseModel


class BaseHigherOrder(BaseModel):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.BaseModel`
    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)

    def _higher_order_macinnes_function(self, x):
        "Function to differentiate between composite and first-order models"
        raise NotImplementedError

    def unpack(self, variables):
        raise NotImplementedError

    def get_coupled_variables(self, variables):
        c_e_av = self.unpack(variables)

        i_boundary_cc_0 = variables["Leading-order current collector current density"]
        c_e = variables["Electrolyte concentration"]
        delta_phi_n_av = variables[
            "X-averaged negative electrode surface potential difference"
        ]
        phi_s_n_av = variables["X-averaged negative electrode potential"]
        tor_n_av = variables["Leading-order x-averaged negative electrolyte tortuosity"]
        tor_s_av = variables["Leading-order x-averaged separator tortuosity"]
        tor_p_av = variables["Leading-order x-averaged positive electrolyte tortuosity"]

        T_av = variables["X-averaged cell temperature"]
        T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        c_e_n, c_e_s, c_e_p = c_e.orphans

        param = self.param
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # bulk conductivities
        kappa_n_av = param.kappa_e(c_e_av, T_av) * tor_n_av
        kappa_s_av = param.kappa_e(c_e_av, T_av) * tor_s_av
        kappa_p_av = param.kappa_e(c_e_av, T_av) * tor_p_av

        chi_av = param.chi(c_e_av)
        if chi_av.domain == ["current collector"]:
            chi_av_n = pybamm.PrimaryBroadcast(chi_av, "negative electrode")
            chi_av_s = pybamm.PrimaryBroadcast(chi_av, "separator")
            chi_av_p = pybamm.PrimaryBroadcast(chi_av, "positive electrode")
        else:
            chi_av_n = chi_av
            chi_av_s = chi_av
            chi_av_p = chi_av

        # electrolyte current
        i_e_n = i_boundary_cc_0 * x_n / l_n
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc_0, "separator")
        i_e_p = i_boundary_cc_0 * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte potential
        phi_e_const = (
            -delta_phi_n_av
            + phi_s_n_av
            - (
                chi_av
                * (1 + param.Theta * T_av)
                * pybamm.x_average(self._higher_order_macinnes_function(c_e_n / c_e_av))
            )
            - (
                (i_boundary_cc_0 * param.C_e * l_n / param.gamma_e)
                * (1 / (3 * kappa_n_av) - 1 / kappa_s_av)
            )
        )

        phi_e_n = (
            phi_e_const
            + (
                chi_av_n
                * (1 + param.Theta * T_av_n)
                * self._higher_order_macinnes_function(c_e_n / c_e_av)
            )
            - (i_boundary_cc_0 * (param.C_e / param.gamma_e) / kappa_n_av)
            * (x_n ** 2 - l_n ** 2)
            / (2 * l_n)
            - i_boundary_cc_0 * l_n * (param.C_e / param.gamma_e) / kappa_s_av
        )

        phi_e_s = (
            phi_e_const
            + (
                chi_av_s
                * (1 + param.Theta * T_av_s)
                * self._higher_order_macinnes_function(c_e_s / c_e_av)
            )
            - (i_boundary_cc_0 * param.C_e / param.gamma_e / kappa_s_av) * x_s
        )

        phi_e_p = (
            phi_e_const
            + (
                chi_av_p
                * (1 + param.Theta * T_av_p)
                * self._higher_order_macinnes_function(c_e_p / c_e_av)
            )
            - (i_boundary_cc_0 * (param.C_e / param.gamma_e) / kappa_p_av)
            * (x_p * (2 - x_p) + l_p ** 2 - 1)
            / (2 * l_p)
            - i_boundary_cc_0 * (1 - l_p) * (param.C_e / param.gamma_e) / kappa_s_av
        )

        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_e_av = pybamm.x_average(phi_e)

        # concentration overpotential
        eta_c_av = (
            chi_av
            * (1 + param.Theta * T_av)
            * (
                pybamm.x_average(self._higher_order_macinnes_function(c_e_p / c_e_av))
                - pybamm.x_average(self._higher_order_macinnes_function(c_e_n / c_e_av))
            )
        )

        # average electrolyte ohmic losses
        delta_phi_e_av = -(param.C_e * i_boundary_cc_0 / param.gamma_e) * (
            param.l_n / (3 * kappa_n_av)
            + param.l_s / (kappa_s_av)
            + param.l_p / (3 * kappa_p_av)
        )

        variables.update(self._get_standard_potential_variables(phi_e, phi_e_av))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables
