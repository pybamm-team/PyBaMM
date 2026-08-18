#
# Class for ambient surface temperature submodel
import pybamm


class Lumped(pybamm.BaseSubModel):
    """
    Class for the lumped surface temperature submodel, which adds an ODE for the
    surface temperature.

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
        T_amb = variables["Ambient temperature [K]"]
        R_env = pybamm.Parameter("Environment thermal resistance [K.W-1]")
        Q_cool_env = -(T_surf - T_amb) / R_env
        variables["Environment total cooling [W]"] = Q_cool_env
        return variables

    def set_rhs(self, variables):
        T_surf = variables["Surface temperature [K]"]

        Q_cool_bulk = variables["Surface total cooling [W]"]
        Q_heat_bulk = -Q_cool_bulk

        Q_cool_env = variables["Environment total cooling [W]"]
        rho_c_p_case = pybamm.Parameter("Casing heat capacity [J.K-1]")

        self.rhs[T_surf] = (Q_heat_bulk + Q_cool_env) / rho_c_p_case

    def set_initial_conditions(self, variables):
        T_surf = variables["Surface temperature [K]"]
        self.initial_conditions = {T_surf: self.param.T_init}
