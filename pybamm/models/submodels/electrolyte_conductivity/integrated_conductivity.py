#
# Composite electrolyte potential employing integrated Stefan-Maxwell
#
import pybamm
from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Integrated(BaseElectrolyteConductivity):
    """
    Integrated model for conservation of charge in the electrolyte derived from
    integrating the Stefan-Maxwell constitutive equations, from [1]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    reactions : dict, optional
        Dictionary of reaction terms

    References
    ----------
    .. [1] F. Brosa Planella, M. Sheikh, and W. D. Widanage, “Systematic derivation and
           validation of reduced thermal-electrochemical models for lithium-ion
           batteries using asymptotic methods.” arXiv preprint, 2020.

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`

    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)
        pybamm.citations.register("BrosaPlanella2020")

    def _higher_order_macinnes_function(self, x):
        return pybamm.log(x)

    def get_coupled_variables(self, variables):
        c_e_av = variables["X-averaged electrolyte concentration"]

        i_boundary_cc_0 = variables["Leading-order current collector current density"]
        c_e_n = variables["Negative electrolyte concentration"]
        c_e_s = variables["Separator electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]
        c_e_n0 = pybamm.boundary_value(c_e_n, "left")

        delta_phi_n_av = variables[
            "X-averaged negative electrode surface potential difference"
        ]
        phi_s_n_av = variables["X-averaged negative electrode potential"]

        tor_n = variables["Negative electrolyte tortuosity"]
        tor_s = variables["Separator tortuosity"]
        tor_p = variables["Positive electrolyte tortuosity"]

        T_av = variables["X-averaged cell temperature"]
        T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        param = self.param
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p
        x_n_edge = pybamm.standard_spatial_vars.x_n_edge
        x_p_edge = pybamm.standard_spatial_vars.x_p_edge

        chi_av = param.chi(c_e_av, T_av)
        chi_av_n = pybamm.PrimaryBroadcast(chi_av, "negative electrode")
        chi_av_s = pybamm.PrimaryBroadcast(chi_av, "separator")
        chi_av_p = pybamm.PrimaryBroadcast(chi_av, "positive electrode")

        # electrolyte current
        i_e_n = i_boundary_cc_0 * x_n / l_n
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc_0, "separator")
        i_e_p = i_boundary_cc_0 * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        i_e_n_edge = i_boundary_cc_0 * x_n_edge / l_n
        i_e_s_edge = pybamm.PrimaryBroadcastToEdges(i_boundary_cc_0, "separator")
        i_e_p_edge = i_boundary_cc_0 * (1 - x_p_edge) / l_p

        # electrolyte potential
        indef_integral_n = (
            pybamm.IndefiniteIntegral(
                i_e_n_edge / (param.kappa_e(c_e_n, T_av_n) * tor_n), x_n
            )
            * param.C_e
            / param.gamma_e
        )
        indef_integral_s = (
            pybamm.IndefiniteIntegral(
                i_e_s_edge / (param.kappa_e(c_e_s, T_av_s) * tor_s), x_s
            )
            * param.C_e
            / param.gamma_e
        )
        indef_integral_p = (
            pybamm.IndefiniteIntegral(
                i_e_p_edge / (param.kappa_e(c_e_p, T_av_p) * tor_p), x_p
            )
            * param.C_e
            / param.gamma_e
        )

        integral_n = indef_integral_n
        integral_s = indef_integral_s + pybamm.boundary_value(integral_n, "right")
        integral_p = indef_integral_p + pybamm.boundary_value(integral_s, "right")

        phi_e_const = (
            -delta_phi_n_av
            + phi_s_n_av
            - (
                chi_av
                * (1 + param.Theta * T_av)
                * pybamm.x_average(self._higher_order_macinnes_function(c_e_n / c_e_n0))
            )
            + pybamm.x_average(integral_n)
        )

        phi_e_n = (
            phi_e_const
            + (
                chi_av_n
                * (1 + param.Theta * T_av_n)
                * self._higher_order_macinnes_function(c_e_n / c_e_n0)
            )
            - integral_n
        )

        phi_e_s = (
            phi_e_const
            + (
                chi_av_s
                * (1 + param.Theta * T_av_s)
                * self._higher_order_macinnes_function(c_e_s / c_e_n0)
            )
            - integral_s
        )

        phi_e_p = (
            phi_e_const
            + (
                chi_av_p
                * (1 + param.Theta * T_av_p)
                * self._higher_order_macinnes_function(c_e_p / c_e_n0)
            )
            - integral_p
        )

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
        delta_phi_e_av = -(pybamm.x_average(integral_p) - pybamm.x_average(integral_n))

        variables.update(
            self._get_standard_potential_variables(phi_e_n, phi_e_s, phi_e_p)
        )
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables
