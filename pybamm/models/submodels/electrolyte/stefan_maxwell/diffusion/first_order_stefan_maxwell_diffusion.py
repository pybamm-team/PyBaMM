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

        # Unpack
        eps_0 = variables["Leading-order porosity"]
        c_e_0 = variables["Leading-order average electrolyte concentration"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]
        deps_0_dt = variables["Leading-order porosity change"]
        dc_e_0_dt = variables["Leading-order electrolyte concentration change"]
        eps_n_0, eps_s_0, eps_p_0 = [e.orphans[0] for e in eps_0.orphans]
        deps_n_0_dt, deps_s_0_dt, deps_p_0_dt = [
            de.orphans[0] for de in deps_0_dt.orphans
        ]

        # Combined time derivatives
        d_epsc_n_0_dt = c_e_0 * deps_n_0_dt + eps_n_0 * dc_e_0_dt
        d_epsc_s_0_dt = eps_s_0 * dc_e_0_dt
        d_epsc_p_0_dt = c_e_0 * deps_p_0_dt + eps_p_0 * dc_e_0_dt

        # Right-hand sides
        D_e_n = (eps_n_0 ** param.b) * param.D_e(c_e_0)
        D_e_s = (eps_s_0 ** param.b) * param.D_e(c_e_0)
        D_e_p = (eps_p_0 ** param.b) * param.D_e(c_e_0)

        # Fluxes

        # Concentrations

        # Update variables
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        variables.update(self._get_standard_concentration_variables(c_e))

        N_e = pybamm.Concatenation(N_e_n, N_e_s, N_e_p)
        variables.update(self._get_standard_flux_variables(N_e))

        return variables
