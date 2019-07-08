#
# Full diffusion limited kinetics
#

import pybamm
from .base_diffusion_limited import BaseModel


class Full(BaseModel):
    """
    Full submodel for diffusion-limited kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_diffusion_limited_current_density(self, variables):
        if self.domain == "Negative":
            import ipdb

            ipdb.set_trace()
            N_ox = variables["Oxygen flux"]

            j = -self.param.l_p * j_p / self.param.l_n

        return j
