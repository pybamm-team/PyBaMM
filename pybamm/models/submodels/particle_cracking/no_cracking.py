#
# Class for cracking
#
import pybamm
from .base_cracking import BaseCracking


class NoSEI(BaseCracking):
    """
    Class for no cracking.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower() + " electrode", "current collector"
        )
        domain = self.domain.lower() + " particle"
        zero_av = pybamm.FullBroadcast(
            pybamm.Scalar(0), "current collector"
        )       
        variables = {
            self.domain + " particle crack length [m]": zero,
            self.domain + " particle crack length": zero,
            f"X-averaged {domain} crack length": zero_av,
            f"X-averaged {domain} crack length [m]": zero_av,
        }
        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower() + " electrode", "current collector"
        )
        variables.update(
            self.domain + " particle surface tangential stress": zero,
            self.domain + " particle surface radial stress": zero,
            self.domain + " particle surface displacement": zero,
            self.domain
            + " particle surface tangential stress [Pa]": zero,
            self.domain + " particle surface radial stress [Pa]": zero,
            self.domain + " particle surface displacement [m]": zero,
        )
        return variables
