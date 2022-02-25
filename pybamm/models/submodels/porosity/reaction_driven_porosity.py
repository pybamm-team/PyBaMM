#
# Class for reaction driven porosity changes as a multiple of SEI/plating thicknesses
#
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
        L_sei_n = variables["Total SEI thickness [m]"]
        L_sei_0 = self.param.L_inner_0_dim + self.param.L_outer_0_dim
        L_pl_n = variables["Lithium plating thickness [m]"]

        L_tot = (L_sei_n - L_sei_0) + L_pl_n

        a_n = variables["Negative electrode surface area to volume ratio [m-1]"]

        # This assumes a thin film so curvature effects are neglected. They could be
        # included (i.e. for a sphere it is a_n * (L_tot + L_tot ** 2 / R_n + L_tot **
        # 3 / (3 * R_n ** 2))) but it is not clear if it is relevant or not.
        delta_eps_n = -a_n * L_tot

        eps_n = self.param.epsilon_n_init + delta_eps_n
        eps_s = self.param.epsilon_s_init
        # no SEI or plating in the positive electrode
        eps_p = self.param.epsilon_p_init

        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)

        return variables
