#
# Lead-acid interface classes
#
import pybamm
from .base_interface import BaseInterface
from . import butler_volmer, inverse_butler_volmer


class BaseModel(BaseInterface, pybamm.lead_acid.BaseModel):
    """
    Base lead-acid interface class

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    and :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_exchange_current_density(self, variables):
        """
        A private function to obtain the exchange current density for a lead acid
        deposition reaction.

        Parameters
        ----------
        variables: dict
        `   The variables in the full model.

        Returns
        -------
        j0 : :class: `pybamm.Symbol`
            The exchange current density.
        """
        c_e = variables[self.domain + " electrolyte concentration"]

        if self.domain == "Negative":
            j0 = self.param.j0_n_S_ref * c_e

        elif self.domain == "Positive":
            c_w = self.param.c_w(c_e)
            j0 = self.param.j0_p_S_ref * c_e ** 2 * c_w

        return j0


class ButlerVolmer(BaseModel, butler_volmer.BaseModel):
    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseButlerVolmer(BaseModel, inverse_butler_volmer.BaseModel):
    def __init__(self, param, domain):
        super().__init__(param, domain)
