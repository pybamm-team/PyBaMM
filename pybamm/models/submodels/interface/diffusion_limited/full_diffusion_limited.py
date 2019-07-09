#
# Full diffusion limited kinetics
#

import pybamm
from .base_diffusion_limited import BaseModel


class BaseFullDiffusionLimited(BaseModel):
    """
    Full submodel for diffusion-limited kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.diffusion_limited.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_diffusion_limited_current_density(self, variables):
        param = self.param
        if self.domain == "Negative":
            N_ox_s_p = variables["Oxygen flux"].orphans[1]
            N_ox_neg_sep_interface = N_ox_s_p[0]

            j = -N_ox_neg_sep_interface / param.C_e / param.s_ox_Ox / param.l_n

        return j
