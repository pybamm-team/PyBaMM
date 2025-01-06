#
# Composite electrolyte potential employing integrated Stefan-Maxwell
#
import pybamm
from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Integrated(BaseElectrolyteConductivity):
    """
    Integrated model for conservation of charge in the electrolyte derived from
    integrating the Stefan-Maxwell constitutive equations, from
    :footcite:t:`BrosaPlanella2021`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)
        pybamm.citations.register("BrosaPlanella2021")

    def _higher_order_macinnes_function(self, x):
        tol = pybamm.settings.tolerances["macinnes__c_e"]
        x = pybamm.maximum(x, tol)
        return pybamm.log(x)

    def build(self, submodels):
        c_e_av = pybamm.CoupledVariable(
            "X-averaged electrolyte concentration [mol.m-3]",
            "current collector",
        )
        self.coupled_variables.update({c_e_av.name: c_e_av})

        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            "current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})

        c_e_n = pybamm.CoupledVariable(
            "Negative electrolyte concentration [mol.m-3]",
            "negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_e_n.name: c_e_n})

        c_e_s = pybamm.CoupledVariable(
            "Separator electrolyte concentration [mol.m-3]",
            "separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_e_s.name: c_e_s})

        c_e_p = pybamm.CoupledVariable(
            "Positive electrolyte concentration [mol.m-3]",
            "positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_e_p.name: c_e_p})

        c_e_n0 = pybamm.boundary_value(c_e_n, "left")

        delta_phi_n_av = pybamm.CoupledVariable(
            "X-averaged negative electrode surface potential difference [V]",
            "current collector",
        )
        self.coupled_variables.update({delta_phi_n_av.name: delta_phi_n_av})

        phi_s_n_av = pybamm.CoupledVariable(
            "X-averaged negative electrode potential [V]",
            "current collector",
        )
        self.coupled_variables.update({phi_s_n_av.name: phi_s_n_av})

        tor_n = pybamm.CoupledVariable(
            "Negative electrolyte transport efficiency",
            "negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor_n.name: tor_n})

        tor_s = pybamm.CoupledVariable(
            "Separator electrolyte transport efficiency",
            "separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor_s.name: tor_s})

        tor_p = pybamm.CoupledVariable(
            "Positive electrolyte transport efficiency",
            "positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor_p.name: tor_p})

        T_av = pybamm.CoupledVariable(
            "X-averaged cell temperature [K]",
            "current collector",
        )
        self.coupled_variables.update({T_av.name: T_av})

        T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        RT_F_av = self.param.R * T_av / self.param.F
        RT_F_av_n = self.param.R * T_av_n / self.param.F
        RT_F_av_s = self.param.R * T_av_s / self.param.F
        RT_F_av_p = self.param.R * T_av_p / self.param.F

        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p
        x_n_edge = pybamm.standard_spatial_vars.x_n_edge
        x_p_edge = pybamm.standard_spatial_vars.x_p_edge

        chi_av = self.param.chi(c_e_av, T_av)
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
            i_e_n_edge / (self.param.kappa_e(c_e_n, T_av_n) * tor_n), x_n
        )
        indef_integral_s = pybamm.IndefiniteIntegral(
            i_e_s_edge / (self.param.kappa_e(c_e_s, T_av_s) * tor_s), x_s
        )
        indef_integral_p = pybamm.IndefiniteIntegral(
            i_e_p_edge / (self.param.kappa_e(c_e_p, T_av_p) * tor_p), x_p
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
        variables = {}
        variables.update(self._get_standard_potential_variables(phi_e_dict))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))
        self.variables.update(variables)
