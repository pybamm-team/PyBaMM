#
# Full diffusion limited kinetics
#
import pybamm
from .full_diffusion_limited import FullDiffusionLimited


class CompositeDiffusionLimited(FullDiffusionLimited):
    """
    Composite submodel for diffusion-limited kinetics.
    Similar to full model, but also adds the first-order x-averaged interfacial current
    density to the dictionary of variables.

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.diffusion_limited.FullDiffusionLimited`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        variables = super().get_coupled_variables(variables)

        j = variables[
            self.domain
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
        ]
        j_0 = variables[
            "Leading-order "
            + self.domain.lower()
            + " electrode"
            + self.reaction_name
            + " interfacial current density"
        ]
        j_1_bar = (pybamm.x_average(j) - pybamm.x_average(j_0)) / self.param.C_e

        variables.update(
            {
                "First-order x-averaged "
                + self.domain.lower()
                + " electrode"
                + self.reaction_name
                + " interfacial current density": j_1_bar
            }
        )

        return variables
