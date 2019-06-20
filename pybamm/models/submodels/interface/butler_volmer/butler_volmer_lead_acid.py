#
# Lead acid Bulter-Volmer class
#

from .base_bulter_volmer import BaseModel


class LeadAcid(BaseModel):
    """
    Lead acid Butler-Volmer class

    *Extends:* :class:`BaseButlerVolmer`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_exchange_current_density(self, variables):
        c_e = variables[self._domain + " electrolyte concentration"]

        if self._domain == "Negative":
            j0 = self.param.m_n * c_e
        elif self._domain == "Positive":
            c_w = self.param.c_w(c_e)
            j0 = self.param.m_p * c_e ** 2 * c_w

        return j0
