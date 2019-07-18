#
# Class for electrolyte diffusion employing stefan-maxwell (first-order)
#
import pybamm
from .base_stefan_maxwell_diffusion import BaseModel


class FirstOrder(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (First-order refers to first-order term in
    asymptotic expansion)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_coupled_variables(self, variables):
        param = self.param
        l_n = param.l_n
        l_s = param.l_s
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack
        eps_0 = variables["Leading-order porosity"]
        c_e_0 = variables["Leading-order average electrolyte concentration"]
        # v_box_0 = variables["Leading-order volume-averaged velocity"]
        deps_0_dt = variables["Leading-order porosity change"]
        dc_e_0_dt = variables["Leading-order electrolyte concentration change"]
        eps_n_0, eps_s_0, eps_p_0 = [e.orphans[0] for e in eps_0.orphans]
        deps_n_0_dt, _, deps_p_0_dt = [de.orphans[0] for de in deps_0_dt.orphans]

        # Combined time derivatives
        d_epsc_n_0_dt = c_e_0 * deps_n_0_dt + eps_n_0 * dc_e_0_dt
        d_epsc_s_0_dt = eps_s_0 * dc_e_0_dt
        d_epsc_p_0_dt = c_e_0 * deps_p_0_dt + eps_p_0 * dc_e_0_dt

        # Right-hand sides
        rhs_n = d_epsc_n_0_dt - sum(
            reaction["Negative"]["s"]
            * variables["Leading-order " + reaction["Negative"]["aj"].lower()].orphans[
                0
            ]
            for reaction in self.reactions.values()
        )
        rhs_s = d_epsc_s_0_dt
        rhs_p = d_epsc_p_0_dt - sum(
            reaction["Positive"]["s"]
            * variables["Leading-order " + reaction["Positive"]["aj"].lower()].orphans[
                0
            ]
            for reaction in self.reactions.values()
        )

        # Diffusivities
        D_e_n = (eps_n_0 ** param.b) * param.D_e(c_e_0)
        D_e_s = (eps_s_0 ** param.b) * param.D_e(c_e_0)
        D_e_p = (eps_p_0 ** param.b) * param.D_e(c_e_0)

        # Fluxes
        N_e_n_1 = -rhs_n * x_n
        N_e_s_1 = -(rhs_s * (x_s - l_n) + rhs_n * l_n)
        N_e_p_1 = -rhs_p * (x_p - 1)

        # Concentrations
        c_e_n_1 = rhs_n / (2 * D_e_n) * (x_n ** 2 - l_n ** 2)
        c_e_s_1 = (rhs_s / (2 * D_e_s) * (x_s - l_n) ** 2) + (
            rhs_n / D_e_s * l_n * (x_s - l_n)
        )
        c_e_p_1 = (
            (rhs_p / (2 * D_e_p) * ((x_p - 1) ** 2 - l_p ** 2))
            + (rhs_s * l_s ** 2 / (2 * D_e_s))
            + (rhs_n * l_n * l_s / D_e_s)
        )

        # Correct for integral
        epsc_e_n_1_av = eps_n_0 * pybamm.Integral(c_e_n_1, x_n)
        epsc_e_s_1_av = eps_s_0 * pybamm.Integral(c_e_s_1, x_s)
        epsc_e_p_1_av = eps_p_0 * pybamm.Integral(c_e_p_1, x_p)
        A_k = -(epsc_e_n_1_av + epsc_e_s_1_av + epsc_e_p_1_av) / (
            l_n * eps_n_0 + l_s * eps_s_0 + l_p * eps_p_0
        )
        c_e_n_1 += A_k
        c_e_s_1 += A_k
        c_e_p_1 += A_k

        # Update variables
        c_e = pybamm.Concatenation(
            pybamm.Broadcast(c_e_0, "negative electrode") + param.C_e * c_e_n_1,
            pybamm.Broadcast(c_e_0, "separator") + param.C_e * c_e_s_1,
            pybamm.Broadcast(c_e_0, "positive electrode") + param.C_e * c_e_p_1,
        )
        variables.update(self._get_standard_concentration_variables(c_e))

        N_e = pybamm.Concatenation(
            param.C_e * N_e_n_1, param.C_e * N_e_s_1, param.C_e * N_e_p_1
        )
        variables.update(self._get_standard_flux_variables(N_e))

        return variables
