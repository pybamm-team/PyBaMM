#
# Class for cracking
#
import pybamm
from .base_mechanics import BaseMechanics
import numpy as np


class CrackPropagation(BaseMechanics):
    """
    Cracking behaviour in electrode particles, from [1]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

    References
    ----------
    .. [1] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
           Battery cycle life prediction with coupled chemical degradation and
           fatigue mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    **Extends:** :class:`pybamm.particle_mechanics.BaseMechanics`
    """

    def __init__(self, param, domain, x_average, options, phase="primary"):
        super().__init__(param, domain, options, phase)
        self.x_average = x_average

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            l_cr_av = pybamm.Variable(
                f"X-averaged {domain} particle crack length",
                domain="current collector",
            )
            l_cr = pybamm.PrimaryBroadcast(l_cr_av, f"{domain} electrode")
        else:
            l_cr = pybamm.Variable(
                f"{Domain} particle crack length",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        return self._get_standard_variables(l_cr)

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        variables.update(self._get_standard_surface_variables(variables))
        variables.update(self._get_mechanical_results(variables))
        T = variables[f"{Domain} electrode temperature"]
        k_cr = self.domain_param.k_cr(T)
        m_cr = self.domain_param.m_cr
        b_cr = self.domain_param.b_cr
        stress_t_surf = variables[f"{Domain} particle surface tangential stress"]
        l_cr = variables[f"{Domain} particle crack length"]
        # # compressive stress will not lead to crack propagation
        dK_SIF = stress_t_surf * b_cr * pybamm.Sqrt(np.pi * l_cr) * (stress_t_surf >= 0)
        dl_cr = k_cr * (dK_SIF**m_cr) / self.param.t0_cr
        variables.update(
            {
                f"{Domain} particle cracking rate": dl_cr,
                f"X-averaged {domain} particle cracking rate": pybamm.x_average(dl_cr),
            }
        )
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            l_cr = variables[f"X-averaged {domain} particle crack length"]
            dl_cr = variables[f"X-averaged {domain} particle cracking rate"]
        else:
            l_cr = variables[f"{Domain} particle crack length"]
            dl_cr = variables[f"{Domain} particle cracking rate"]
        self.rhs = {l_cr: dl_cr}

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            l_cr = variables[f"X-averaged {domain} particle crack length"]
            l0 = 1
        else:
            l_cr = variables[f"{Domain} particle crack length"]
            l0 = pybamm.PrimaryBroadcast(1, f"{domain} electrode")
        self.initial_conditions = {l_cr: l0}

    def set_events(self, variables):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            l_cr = variables[f"X-averaged {domain} particle crack length"]
        else:
            l_cr = variables[f"{Domain} particle crack length"]
        self.events.append(
            pybamm.Event(
                f"{domain} particle crack length larger than particle radius",
                self.domain_param.prim.R_typ / self.domain_param.l_cr_0
                - pybamm.max(l_cr),
                pybamm.EventType.TERMINATION,
            )
        )
