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
    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)

    def _derivative_macinnes_function(self, x):
        "Compute the derivative of the MacInnes function."
        tol = pybamm.settings.tolerances["macinnes__c_e"]
        x = pybamm.maximum(x, tol)
        return 1 / x

    def get_coupled_variables(self, variables):
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        if self.options.electrode_types["negative"] == "porous":
            c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
            delta_phi_n_av = variables[
                "X-averaged negative electrode surface potential difference [V]"
            ]
            phi_s_n_av = variables["X-averaged negative electrode potential [V]"]
            tor_n = variables["Negative electrolyte transport efficiency"]

        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        tor_s = variables["Separator electrolyte transport efficiency"]
        tor_p = variables["Positive electrolyte transport efficiency"]

        T_av = variables["X-averaged cell temperature [K]"]
        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        param = self.param
        RT_F_av_s = param.R * T_av_s / param.F
        RT_F_av_p = param.R * T_av_p / param.F

        L_n = param.n.L
        L_p = param.p.L
        L_x = param.L_x
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p
        x_p_edge = pybamm.standard_spatial_vars.x_p_edge

        # electrolyte current
        if self.options.electrode_types["negative"] == "planar":
            # i_e_n = None
            pass
        else:
            x_n = pybamm.standard_spatial_vars.x_n
            x_n_edge = pybamm.standard_spatial_vars.x_n_edge
            T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
            RT_F_av_n = param.R * T_av_n / param.F
            # i_e_n = i_boundary_cc * x_n / L_n
            i_e_n_edge = i_boundary_cc * x_n_edge / L_n
        # i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
        i_e_s_edge = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")
        # i_e_p = i_boundary_cc * (L_x - x_p) / L_p
        i_e_p_edge = i_boundary_cc * (L_x - x_p_edge) / L_p
        i_e = pybamm.concatenation(i_e_n_edge, i_e_s_edge, i_e_p_edge)
        phi_e_dict = {}

        # electrolyte potential
        if self.options.electrode_types["negative"] == "planar":
            phi_e_const = variables["Lithium metal interface electrolyte potential [V]"]
            eta_c_n = pybamm.Scalar(0)
            delta_phi_e_n = pybamm.Scalar(0)
        else:
            # need to have current evaluated on edge as integral will make it
            # evaluate on node

            eta_c_n = -RT_F_av_n * pybamm.IndefiniteIntegral(
                param.chi(c_e_n, T_av_n)
                * self._derivative_macinnes_function(c_e_n)
                * pybamm.grad(c_e_n),
                x_n,
            )

            delta_phi_e_n = pybamm.IndefiniteIntegral(
                i_e_n_edge / (param.kappa_e(c_e_n, T_av_n) * tor_n), x_n
            )
            phi_e_const = (
                -delta_phi_n_av
                + phi_s_n_av
                + pybamm.x_average(eta_c_n)
                + pybamm.x_average(delta_phi_e_n)
            )

            phi_e_n = phi_e_const - (delta_phi_e_n + eta_c_n)
            phi_e_dict["negative electrode"] = phi_e_n

        eta_c_s = -RT_F_av_s * pybamm.IndefiniteIntegral(
            param.chi(c_e_s, T_av_s)
            * self._derivative_macinnes_function(c_e_s)
            * pybamm.grad(c_e_s),
            x_s,
        )
        delta_phi_e_s = pybamm.IndefiniteIntegral(
            i_e_s_edge / (param.kappa_e(c_e_s, T_av_s) * tor_s), x_s
        )

        phi_e_s = (
            phi_e_const
            - pybamm.boundary_value(delta_phi_e_n, "right")
            - pybamm.boundary_value(eta_c_n, "right")
            - (delta_phi_e_s + eta_c_s)
        )

        eta_c_p = -RT_F_av_p * pybamm.IndefiniteIntegral(
            param.chi(c_e_p, T_av_p)
            * self._derivative_macinnes_function(c_e_p)
            * pybamm.grad(c_e_p),
            x_p,
        )
        delta_phi_e_p = pybamm.IndefiniteIntegral(
            i_e_p_edge / (param.kappa_e(c_e_p, T_av_p) * tor_p), x_p
        )

        phi_e_p = (
            phi_e_const
            - pybamm.boundary_value(delta_phi_e_n, "right")
            - pybamm.boundary_value(eta_c_n, "right")
            - pybamm.boundary_value(delta_phi_e_s, "right")
            - pybamm.boundary_value(eta_c_s, "right")
            - (delta_phi_e_p + eta_c_p)
        )

        phi_e_dict["separator"] = phi_e_s
        phi_e_dict["positive electrode"] = phi_e_p

        # concentration overpotential
        eta_c_av = (
            -pybamm.x_average(eta_c_p)
            + pybamm.x_average(eta_c_n)
            - pybamm.boundary_value(eta_c_s, "right")
            - pybamm.boundary_value(eta_c_n, "right")
        )

        # average electrolyte ohmic losses
        delta_phi_e_av = (
            -pybamm.x_average(delta_phi_e_p)
            + pybamm.x_average(delta_phi_e_n)
            - pybamm.boundary_value(delta_phi_e_s, "right")
            - pybamm.boundary_value(delta_phi_e_n, "right")
        )

        variables.update(self._get_standard_potential_variables(phi_e_dict))
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        # Override print_name
        i_e.print_name = "i_e"

        return variables

    def set_boundary_conditions(self, variables):
        # Define the boundary conditions for electrolyte concentration and potential
        # so the gradients can be computed
        if self.options.electrode_types["negative"] == "planar":
            domains = ["Separator", "Positive"]
        else:
            domains = ["Negative", "Separator", "Positive"]

        var_names = [
            " electrode porosity times concentration [mol.m-3]",
            " electrolyte concentration [mol.m-3]",
            " electrolyte potential [V]",
        ]

        for name in var_names:
            for domain in domains:
                full_var_name = domain + name

                # Fix inconcsistency in naming, where porosity times concentration is
                # called "electrode porosity times concentration"
                full_var_name = full_var_name.replace(
                    "Separator electrode", "Separator"
                )
                var = variables[full_var_name]
                self.boundary_conditions.update(
                    {
                        var: {
                            "left": (pybamm.boundary_gradient(var, "left"), "Neumann"),
                            "right": (
                                pybamm.boundary_gradient(var, "right"),
                                "Neumann",
                            ),
                        },
                    }
                )
