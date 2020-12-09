#
# Class for cracking
#
import pybamm
from .base_cracking import BaseCracking
import numpy as np


class CrackPropagation(BaseCracking):
    """
    Cracking behaviour in electrode particles, from [1]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    requiring the radius, average concantration, surface concantration

    References
    ----------
    .. [1] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
           Battery cycle life prediction with coupled chemical degradation and
           fatigue mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    **Extends:** :class:`pybamm.particle_cracking.BaseCracking`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        l_cr = pybamm.Variable(
            self.domain + " particle crack length",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        return self._get_standard_variables(l_cr)

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_surface_variables(variables))
        variables.update(self._get_mechanical_results(variables))
        T = variables[self.domain + " electrode temperature"]
        if self.domain == "Negative":
            k_cr = self.param.k_cr_n(T)
            m_cr = self.param.m_cr_n
            b_cr = self.param.b_cr_n
        else:
            k_cr = self.param.k_cr_p(T)
            m_cr = self.param.m_cr_p
            b_cr = self.param.b_cr_p
        stress_t_surf = variables[self.domain + " particle surface tangential stress"]
        l_cr = variables[self.domain + " particle crack length"]
        # # compressive stress will not lead to crack propagation
        dK_SIF = stress_t_surf * b_cr * pybamm.Sqrt(np.pi * l_cr) * (stress_t_surf >= 0)
        dl_cr = k_cr * pybamm.Power(dK_SIF, m_cr) / self.param.t0_cr
        variables.update(
            {
                self.domain + " particle cracking rate": dl_cr,
                "X-averaged "
                + self.domain.lower()
                + " particle cracking rate": pybamm.x_average(dl_cr),
            }
        )
        return variables

    def set_rhs(self, variables):
        l_cr = variables[self.domain + " particle crack length"]
        dl_cr = variables[self.domain + " particle cracking rate"]
        self.rhs = {l_cr: dl_cr}

    def set_initial_conditions(self, variables):
        l_cr = variables[self.domain + " particle crack length"]
        l0 = pybamm.PrimaryBroadcast(
            pybamm.Scalar(1), [self.domain.lower() + " electrode"]
        )
        self.initial_conditions = {l_cr: l0}
