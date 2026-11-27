#
# Class for lumped surface temperature submodel
import pybamm


class Lumped(pybamm.BaseSubModel):
    """
    Class for the lumped surface temperature submodel, which adds an ODE for the
    surface temperature. Heat reaches the surface from the cell core through the
    core-to-surface thermal resistance, and leaves to the ambient by Newton cooling
    with the total heat transfer coefficient. For more information see
    :footcite:t:`Lin2014`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Lin2014")

    def get_fundamental_variables(self):
        T_surf = pybamm.Variable("Surface temperature [K]")
        variables = {"Surface temperature [K]": T_surf}

        return variables

    def get_coupled_variables(self, variables):
        T_surf = variables["Surface temperature [K]"]
        T_amb = variables["Volume-averaged ambient temperature [K]"]
        Q_surface_env = self.param.h_total * self.param.A_cooling * (T_surf - T_amb)
        variables["Surface to environment heat flux [W]"] = Q_surface_env
        return variables

    def set_rhs(self, variables):
        T_surf = variables["Surface temperature [K]"]

        Q_core_surface = variables["Core to surface heat flux [W]"]
        Q_surface_env = variables["Surface to environment heat flux [W]"]
        rho_c_p_case = pybamm.Parameter("Casing heat capacity [J.K-1]")

        self.rhs[T_surf] = (Q_core_surface - Q_surface_env) / rho_c_p_case

    def set_initial_conditions(self, variables):
        T_surf = variables["Surface temperature [K]"]
        self.initial_conditions = {T_surf: self.param.T_init}
