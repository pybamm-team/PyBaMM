#
# Base class for electrolyte diffusion employing stefan-maxwell
#
import pybamm


class BaseStefanMaxwellDiffusion(pybamm.BaseSubModel):
    """Base class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

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

    def _get_standard_flux_variables(self, N_e, N_e_av):

        param = self.param
        D_e_typ = param.D_e(param.c_e_typ)
        flux_scale = D_e_typ * param.c_e_typ / param.L_x

        variables = {
            "Electrolyte flux": N_e,
            "Electrolyte flux [mol.m-2.s-1]": N_e * flux_scale,
            "Average electrolyte flux": pybamm.average(N_e),
            "Average electrolyte flux [mol.m-2.s-1]": pybamm.average(N_e) * flux_scale,
        }

        return variables

