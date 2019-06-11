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

    def _flux_law(self, epsilon, c_e, i_e, v_box):
        param = self.param

        N_e_diffusion = -(epsilon ** param.b) * param.D_e(c_e) * pybamm.grad(c_e)
        N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        N_e_convection = c_e * v_box

        return N_e_diffusion + N_e_migration + N_e_convection

