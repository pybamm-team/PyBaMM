#
# Class for reaction driven porosity changes as a multiple of SEI/plating thicknesses
#
import pybamm
from .base_porosity import BaseModel


class ReactionDriven(BaseModel):
    """Reaction-driven porosity changes as a multiple of SEI/plating thicknesses

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        Options dictionary passed from the full model
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)

    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def __init__(self, param, options, x_average):
        super().__init__(param, options)
        self.x_average = x_average

    def get_coupled_variables(self, variables):
        L_sei_n = variables["Negative electrode SEI thickness"]
        L_sei_0 = self.param.L_inner_0 + self.param.L_outer_0
        L_pl_n = variables["Negative electrode lithium plating thickness"]

        eps_n = self.param.epsilon_n_init - (L_sei_n - L_sei_0) - L_pl_n
        eps_s = pybamm.FullBroadcast(
            self.param.epsilon_s_init, "separator", "current collector"
        )
        # no SEI or plating in the positive electrode
        eps_p = pybamm.FullBroadcast(
            self.param.epsilon_p_init, "positive electrode", "current collector"
        )
        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)

        return variables
