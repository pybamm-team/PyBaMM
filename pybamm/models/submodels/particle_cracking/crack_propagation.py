import pybamm
from .base_cracking import BaseCracking
import numpy as np


class CrackPropagation(BaseCracking):
    """cracking behaviour in electrode particles.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    requiring the radius, average concantration, surface concantration

    Ref for the crack model: Deshpande, R., Verbrugge, M., Cheng, Y. T.,
    Wang, J., & Liu, P. (2012). Battery cycle life prediction with coupled
    chemical degradation and fatigue mechanics. Journal of the Electrochemical
    Society, 159(10), A1730.
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        l_cr = pybamm.Variable(
            self.domain + " particle crack length",
            domain=self.domain.lower() + " electrode",
        )
        return self._get_standard_variables(l_cr)

    def get_coupled_variables(self, variables):
        variables.update(self._get_mechanical_results(variables))
        if self.domain == "Negative":
            l_cr_scale = self.param.l_cr_n_0
            Eac_cr = self.param.Eac_cr_n
        else:
            l_cr_scale = self.param.l_cr_p_0
            Eac_cr = self.param.Eac_cr_p
        stress_t_surf = variables[
            self.domain + " particle surface tangential stress [Pa]"
        ]
        l_cr = variables[self.domain + " particle crack length"]
        R_const = self.param.R_const
        Delta_T = self.param.Delta_T
        T_dim = variables[self.domain + " electrode temperature [K]"]
        k_cr = self.param.k_cr * pybamm.exp(
            Eac_cr / R_const * (1 / T_dim - 1 / self.param.T_ref)
        )  # cracking rate with temperature dependence
        # # compressive stress will not lead to crack propagation
        dK_SIF = (
            stress_t_surf
            * self.param.b_cr
            * pybamm.Sqrt(np.pi * l_cr * l_cr_scale)
            * (stress_t_surf >= 0)
        )
        dl_cr = (
            k_cr * pybamm.Power(dK_SIF, self.param.m_cr) / self.param.t0_cr / l_cr_scale
        )
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
