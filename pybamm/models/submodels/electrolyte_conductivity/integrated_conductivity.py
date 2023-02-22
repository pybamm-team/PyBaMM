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
    options : dict, optional
        A dictionary of options to be passed to the model.

    References
    ----------
    .. [1] F. Brosa Planella, M. Sheikh, and W. D. Widanage, “Systematic derivation and
           validation of reduced thermal-electrochemical models for lithium-ion
           batteries using asymptotic methods.” arXiv preprint, 2020.

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`

    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)
        pybamm.citations.register("BrosaPlanella2021")

    def _higher_order_macinnes_function(self, x):
        tol = pybamm.settings.tolerances["macinnes__c_e"]
        x = pybamm.maximum(x, tol)
        return pybamm.log(x)

    def get_coupled_variables(self, variables):
        param = self.param
        c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]

        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]
        c_e_n0 = pybamm.boundary_value(c_e_n, "left")

        delta_phi_n_av = variables[
            "X-averaged negative electrode surface potential difference [V]"
        ]
        phi_s_n_av = variables["X-averaged negative electrode potential [V]"]

        tor_n = variables["Negative electrolyte transport efficiency"]
        tor_s = variables["Separator electrolyte transport efficiency"]
        tor_p = variables["Positive electrolyte transport efficiency"]

        T_av = variables["X-averaged cell temperature [K]"]
        T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        RT_F_av = param.R * T_av / param.F
        RT_F_av_n = param.R * T_av_n / param.F
        RT_F_av_s = param.R * T_av_s / param.F
        RT_F_av_p = param.R * T_av_p / param.F

        param = self.param
        L_n = param.n.L
        L_p = param.p.L
        L_x = param.L_x
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
        i_e_n = i_boundary_cc * x_n / L_n
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
        i_e_p = i_boundary_cc * (L_x - x_p) / L_p
        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)

        i_e_n_edge = i_boundary_cc * x_n_edge / L_n
        i_e_s_edge = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")
        i_e_p_edge = i_boundary_cc * (L_x - x_p_edge) / L_p

        # electrolyte potential
        indef_integral_n = pybamm.IndefiniteIntegral(
            i_e_n_edge / (param.kappa_e(c_e_n, T_av_n) * tor_n), x_n
        )
        indef_integral_s = pybamm.IndefiniteIntegral(
            i_e_s_edge / (param.kappa_e(c_e_s, T_av_s) * tor_s), x_s
        )
        indef_integral_p = pybamm.IndefiniteIntegral(
            i_e_p_edge / (param.kappa_e(c_e_p, T_av_p) * tor_p), x_p
        )

        integral_n = indef_integral_n
        integral_s = indef_integral_s + pybamm.boundary_value(integral_n, "right")
        integral_p = indef_integral_p + pybamm.boundary_value(integral_s, "right")

        phi_e_const = (
            -delta_phi_n_av
            + phi_s_n_av
            - (
                chi_av
                * RT_F_av
                * pybamm.x_average(self._higher_order_macinnes_function(c_e_n / c_e_n0))
            )
            + pybamm.x_average(integral_n)
        )

        phi_e_n = (
            phi_e_const
            + (
                chi_av_n
                * RT_F_av_n
                * self._higher_order_macinnes_function(c_e_n / c_e_n0)
            )
            - integral_n
        )

        phi_e_s = (
            phi_e_const
            + (
                chi_av_s
                * RT_F_av_s
                * self._higher_order_macinnes_function(c_e_s / c_e_n0)
            )
            - integral_s
        )

        phi_e_p = (
            phi_e_const
            + (
                chi_av_p
                * RT_F_av_p
                * self._higher_order_macinnes_function(c_e_p / c_e_n0)
            )
            - integral_p
        )

        # concentration overpotential
        eta_c_av = (
            chi_av
            * RT_F_av
            * (
                pybamm.x_average(self._higher_order_macinnes_function(c_e_p / c_e_av))
                - pybamm.x_average(self._higher_order_macinnes_function(c_e_n / c_e_av))
            )
        )

        # average electrolyte ohmic losses
        delta_phi_e_av = -(pybamm.x_average(integral_p) - pybamm.x_average(integral_n))

        phi_e_dict = {
            "negative electrode": phi_e_n,
            "separator": phi_e_s,
            "positive electrode": phi_e_p,
        }
        variables.update(self._get_standard_potential_variables(phi_e_dict))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables
