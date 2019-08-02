#
# Leading-order diffusion limited kinetics
#
from .base_diffusion_limited import BaseModel


class LeadingOrderDiffusionLimited(BaseModel):
    """
    Leading-order submodel for diffusion-limited kinetics

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
        if self.domain == "Negative":
            j_p = variables[
                "X-averaged positive electrode"
                + self.reaction_name
                + " interfacial current density"
            ]
            j = -self.param.l_p * j_p / self.param.l_n

        return j

    def _get_j_diffusion_limited_first_order(self, variables):
        """
        First-order correction to the interfacial current density due to
        diffusion-limited effects. For a general model the correction term is zero,
        since the reaction is not diffusion-limited
        """
        j_leading_order = variables[
            "Leading-order x-averaged "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
        ]
        param = self.param
        if self.domain == "Negative":
            N_ox_s_p = variables["Oxygen flux"].orphans[1]
            N_ox_neg_sep_interface = N_ox_s_p[0]

            j = -N_ox_neg_sep_interface / param.C_e / param.s_ox_Ox / param.l_n

        return (j - j_leading_order) / param.C_e
