#
# Base class for electrolyte diffusion
#
import pybamm


class BaseElectrolyteDiffusion(pybamm.BaseSubModel):
    """Base class for conservation of mass in the electrolyte.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, ocp=False):
        super().__init__(param)
        self.ocp = ocp

    def _get_standard_concentration_variables(self, c_e, c_e_av):

        c_e_typ = self.param.c_e_typ
        c_e_n, c_e_s, c_e_p = c_e.orphans

        variables = {
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e_typ * c_e,
            "Average electrolyte concentration": c_e_av,
            "Average electrolyte concentration [mol.m-3]": c_e_typ * c_e_av,
            "Negative electrolyte concentration": c_e_n,
            "Negative electrolyte concentration [mol.m-3]": c_e_typ * c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Separator electrolyte concentration [mol.m-3]": c_e_typ * c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "Positive electrolyte concentration [mol.m-3]": c_e_typ * c_e_p,
        }

        return variables

    def _get_standard_flux_variables(self, N_e):

        param = self.param
        D_e_typ = param.D_e(param.c_e_typ)
        flux_scale = D_e_typ * param.c_e_typ / param.L_x

        variables = {
            "Electrolyte flux": N_e,
            "Electrolyte flux [mol.m-2.s-1]": N_e * flux_scale,
        }

        return variables

    def set_events(self, variables):
        c_e = variables["Electrolyte concentration"]
        self.events["Zero electrolyte concentration cut-off"] = pybamm.min(c_e) - 0.002

    def _get_standard_ocp_variables(self, c_e):

        c_e_n, _, c_e_p = c_e.orphans

        ocp_n = self.param.U_n(c_e_n)
        ocp_p = self.param.U_p(c_e_p)

        ocp_n_dim = self.param.U_n_ref + self.param.potential_scale * ocp_n
        ocp_p_dim = self.param.U_p_ref + self.param.potential_scale * ocp_p

        ocp_n_av = pybamm.average(ocp_n)
        ocp_n_av_dim = pybamm.average(ocp_n_dim)

        ocp_p_av = pybamm.average(ocp_p)
        ocp_p_av_dim = pybamm.average(ocp_p_dim)

        variables = {
            "Negative electrode open circuit potential": ocp_n,
            "Negative electrode open circuit potential [V]": ocp_n_dim,
            "Average negative electrode open circuit potential": ocp_n_av,
            "Average negative electrode open circuit potential [V]": ocp_n_av_dim,
            "Positive electrode open circuit potential": ocp_p,
            "Positive electrode open circuit potential [V]": ocp_p_dim,
            "Average positive electrode open circuit potential": ocp_p_av,
            "Average positive electrode open circuit potential [V]": ocp_p_av_dim,
        }

        return variables

