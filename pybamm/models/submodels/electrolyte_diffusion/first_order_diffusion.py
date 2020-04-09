#
# Class for electrolyte diffusion employing stefan-maxwell (first-order)
#
import pybamm
from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class FirstOrder(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (First-order refers to first-order term in
    asymptotic expansion)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):
        param = self.param
        l_n = param.l_n
        l_s = param.l_s
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack
        T_0 = variables["Leading-order cell temperature"]
        c_e_0 = variables["Leading-order x-averaged electrolyte concentration"]
        # v_box_0 = variables["Leading-order volume-averaged velocity"]
        dc_e_0_dt = variables["Leading-order electrolyte concentration change"]
        eps_n_0 = variables["Leading-order x-averaged negative electrode porosity"]
        eps_s_0 = variables["Leading-order x-averaged separator porosity"]
        eps_p_0 = variables["Leading-order x-averaged positive electrode porosity"]
        tor_n_0 = variables["Leading-order x-averaged negative electrolyte tortuosity"]
        tor_s_0 = variables["Leading-order x-averaged separator tortuosity"]
        tor_p_0 = variables["Leading-order x-averaged positive electrolyte tortuosity"]
        deps_n_0_dt = variables[
            "Leading-order x-averaged negative electrode porosity change"
        ]
        deps_p_0_dt = variables[
            "Leading-order x-averaged positive electrode porosity change"
        ]

        # Combined time derivatives
        d_epsc_n_0_dt = c_e_0 * deps_n_0_dt + eps_n_0 * dc_e_0_dt
        d_epsc_s_0_dt = eps_s_0 * dc_e_0_dt
        d_epsc_p_0_dt = c_e_0 * deps_p_0_dt + eps_p_0 * dc_e_0_dt

        # Right-hand sides
        sum_j_n_0 = variables[
            "Leading-order sum of x-averaged "
            "negative electrode interfacial current densities"
        ]
        sum_j_p_0 = variables[
            "Leading-order sum of x-averaged "
            "positive electrode interfacial current densities"
        ]
        sum_s_j_n_0 = variables[
            "Leading-order sum of x-averaged "
            "negative electrode electrolyte reaction source terms"
        ]
        sum_s_j_p_0 = variables[
            "Leading-order sum of x-averaged "
            "positive electrode electrolyte reaction source terms"
        ]
        rhs_n = (
            d_epsc_n_0_dt
            - (sum_s_j_n_0 - param.t_plus(c_e_0) * sum_j_n_0) / param.gamma_e
        )
        rhs_s = d_epsc_s_0_dt
        rhs_p = (
            d_epsc_p_0_dt
            - (sum_s_j_p_0 - param.t_plus(c_e_0) * sum_j_p_0) / param.gamma_e
        )

        # Diffusivities
        D_e_n = tor_n_0 * param.D_e(c_e_0, T_0)
        D_e_s = tor_s_0 * param.D_e(c_e_0, T_0)
        D_e_p = tor_p_0 * param.D_e(c_e_0, T_0)

        # Fluxes
        N_e_n_1 = -rhs_n * x_n
        N_e_s_1 = -(rhs_s * (x_s - l_n) + rhs_n * l_n)
        N_e_p_1 = -rhs_p * (x_p - 1)

        # Concentrations
        c_e_n_1 = (rhs_n / (2 * D_e_n)) * (x_n ** 2 - l_n ** 2)
        c_e_s_1 = (rhs_s / 2) * ((x_s - l_n) ** 2) + (rhs_n * l_n / D_e_s) * (x_s - l_n)
        c_e_p_1 = (rhs_p / (2 * D_e_p)) * ((x_p - 1) ** 2 - l_p ** 2) + (
            (rhs_s * l_s ** 2 / (2 * D_e_s)) + (rhs_n * l_n * l_s / D_e_s)
        )

        # Correct for integral
        c_e_n_1_av = -rhs_n * l_n ** 3 / (3 * D_e_n)
        c_e_s_1_av = (rhs_s * l_s ** 3 / 6 + rhs_n * l_n * l_s ** 2 / 2) / D_e_s
        c_e_p_1_av = (
            -rhs_p * l_p ** 3 / (3 * D_e_p)
            + (rhs_s * l_s ** 2 * l_p / (2 * D_e_s))
            + (rhs_n * l_n * l_s * l_p / D_e_s)
        )
        A_e = -(eps_n_0 * c_e_n_1_av + eps_s_0 * c_e_s_1_av + eps_p_0 * c_e_p_1_av) / (
            l_n * eps_n_0 + l_s * eps_s_0 + l_p * eps_p_0
        )
        c_e_n_1 += A_e
        c_e_s_1 += A_e
        c_e_p_1 += A_e
        c_e_n_1_av += A_e
        c_e_s_1_av += A_e
        c_e_p_1_av += A_e

        # Update variables
        c_e_n = c_e_0 + param.C_e * c_e_n_1
        c_e_s = c_e_0 + param.C_e * c_e_s_1
        c_e_p = c_e_0 + param.C_e * c_e_p_1
        variables.update(
            self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)
        )
        # Update with analytical expressions for first-order x-averages
        variables.update(
            {
                "X-averaged first-order negative electrolyte concentration": c_e_n_1_av,
                "X-averaged first-order separator concentration": c_e_s_1_av,
                "X-averaged first-order positive electrolyte concentration": c_e_p_1_av,
            }
        )

        N_e = pybamm.Concatenation(
            param.C_e * N_e_n_1, param.C_e * N_e_s_1, param.C_e * N_e_p_1
        )
        variables.update(self._get_standard_flux_variables(N_e))

        return variables
