#
# Composite electrolyte potential employing stefan-maxwell
#
import pybamm
from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Composite(BaseElectrolyteConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.
    higher_order_terms : str
        What kind of higher-order terms to use ('composite' or 'first-order')

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)

    def _higher_order_macinnes_function(self, x):
        "Function to differentiate between composite and first-order models"
        tol = pybamm.settings.tolerances["macinnes__c_e"]
        x = pybamm.maximum(x, tol)
        return pybamm.log(x)

    def get_coupled_variables(self, variables):
        c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]

        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        if self.options.electrode_types["negative"] == "porous":
            c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
            delta_phi_n_av = variables[
                "X-averaged negative electrode surface potential difference [V]"
            ]
            phi_s_n_av = variables["X-averaged negative electrode potential [V]"]
            tor_n_av = variables["X-averaged negative electrolyte transport efficiency"]

        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        tor_s_av = variables["X-averaged separator electrolyte transport efficiency"]
        tor_p_av = variables["X-averaged positive electrolyte transport efficiency"]

        T_av = variables["X-averaged cell temperature [K]"]
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        param = self.param
        RT_F_av = param.R * T_av / param.F
        RT_F_av_s = param.R * T_av_s / param.F
        RT_F_av_p = param.R * T_av_p / param.F

        L_n = param.n.L
        L_s = param.s.L
        L_p = param.p.L
        L_x = param.L_x
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # bulk conductivities
        kappa_s_av = param.kappa_e(c_e_av, T_av) * tor_s_av
        kappa_p_av = param.kappa_e(c_e_av, T_av) * tor_p_av

        chi_av = param.chi(c_e_av, T_av)
        chi_av_s = pybamm.PrimaryBroadcast(chi_av, "separator")
        chi_av_p = pybamm.PrimaryBroadcast(chi_av, "positive electrode")

        # electrolyte current
        if self.options.electrode_types["negative"] == "planar":
            i_e_n = None
        else:
            x_n = pybamm.standard_spatial_vars.x_n
            chi_av_n = pybamm.PrimaryBroadcast(chi_av, "negative electrode")
            T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
            RT_F_av_n = param.R * T_av_n / param.F
            kappa_n_av = param.kappa_e(c_e_av, T_av) * tor_n_av
            i_e_n = i_boundary_cc * x_n / L_n
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
        i_e_p = i_boundary_cc * (L_x - x_p) / L_p
        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)

        phi_e_dict = {}

        # electrolyte potential
        if self.options.electrode_types["negative"] == "planar":
            phi_e_li = variables["Lithium metal interface electrolyte potential [V]"]
            c_e_n = pybamm.boundary_value(c_e_s, "left")
            phi_e_const = (
                phi_e_li
                - chi_av
                * RT_F_av
                * self._higher_order_macinnes_function(c_e_n / c_e_av)
                + (i_boundary_cc / kappa_s_av) * L_n
            )
        else:
            phi_e_const = (
                -delta_phi_n_av
                + phi_s_n_av
                - (
                    chi_av
                    * RT_F_av
                    * pybamm.x_average(
                        self._higher_order_macinnes_function(c_e_n / c_e_av)
                    )
                )
                - ((i_boundary_cc * L_n) * (1 / (3 * kappa_n_av) - 1 / kappa_s_av))
            )

            phi_e_n = (
                phi_e_const
                + (
                    chi_av_n
                    * RT_F_av_n
                    * self._higher_order_macinnes_function(c_e_n / c_e_av)
                )
                - (i_boundary_cc / kappa_n_av) * (x_n**2 - L_n**2) / (2 * L_n)
                - i_boundary_cc * L_n / kappa_s_av
            )
            phi_e_dict["negative electrode"] = phi_e_n

        phi_e_s = (
            phi_e_const
            + (
                chi_av_s
                * RT_F_av_s
                * self._higher_order_macinnes_function(c_e_s / c_e_av)
            )
            - (i_boundary_cc / kappa_s_av) * x_s
        )

        phi_e_p = (
            phi_e_const
            + (
                chi_av_p
                * RT_F_av_p
                * self._higher_order_macinnes_function(c_e_p / c_e_av)
            )
            - (i_boundary_cc / kappa_p_av)
            * (x_p * (2 * L_x - x_p) + L_p**2 - L_x**2)
            / (2 * L_p)
            - i_boundary_cc * (L_x - L_p) / kappa_s_av
        )

        phi_e_dict["separator"] = phi_e_s
        phi_e_dict["positive electrode"] = phi_e_p

        # concentration overpotential
        macinnes_c_e_p = pybamm.x_average(
            self._higher_order_macinnes_function(c_e_p / c_e_av)
        )
        if self.options.electrode_types["negative"] == "planar":
            macinnes_c_e_n = 0
            ohmic_n = 0
        else:
            macinnes_c_e_n = pybamm.x_average(
                self._higher_order_macinnes_function(c_e_n / c_e_av)
            )
            ohmic_n = L_n / (3 * kappa_n_av)

        eta_c_av = chi_av * RT_F_av * (macinnes_c_e_p - macinnes_c_e_n)

        # average electrolyte ohmic losses
        delta_phi_e_av = -(i_boundary_cc) * (
            ohmic_n + L_s / kappa_s_av + L_p / (3 * kappa_p_av)
        )

        variables.update(self._get_standard_potential_variables(phi_e_dict))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        # Override print_name
        i_e.print_name = "i_e"

        return variables
