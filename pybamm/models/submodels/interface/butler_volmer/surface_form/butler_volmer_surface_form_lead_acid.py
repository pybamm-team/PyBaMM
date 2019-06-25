#
# Lead acid Bulter-Volmer class
#

from .base_surface_form_butler_volmer import BaseModel


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
            j0 = self.param.j0_n_S_ref * c_e
        elif self._domain == "Positive":
            c_w = self.param.c_w(c_e)
            j0 = self.param.j0_p_S_ref * c_e ** 2 * c_w

        return j0
