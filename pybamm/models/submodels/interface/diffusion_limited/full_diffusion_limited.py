#
# Full diffusion limited kinetics
#
import pybamm
from .base_diffusion_limited import BaseModel


class FullDiffusionLimited(BaseModel):
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
            tor_s = variables["Separator tortuosity"]
            c_ox_s = variables["Separator oxygen concentration"]
            N_ox_neg_sep_interface = (
                -pybamm.boundary_value(tor_s, "left")
                * param.curlyD_ox
                * pybamm.BoundaryGradient(c_ox_s, "left")
            )
            N_ox_neg_sep_interface.domain = ["current collector"]

            j = -N_ox_neg_sep_interface / param.C_e / param.s_ox_Ox / param.l_n

        return j
