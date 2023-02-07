import pybamm


class ThermalSubModel(pybamm.BaseSubModel):
    """
    Thermal SubModel for use with equivalent
    circuits.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        self.model_options = options

    def get_fundamental_variables(self):
        T_cell = pybamm.Variable("Cell temperature [degC]")
        T_jig = pybamm.Variable("Jig temperature [degC]")

        T_amb = self.param.T_amb(pybamm.t * self.param.timescale)

        Q_cell_cool = -self.param.k_cell_jig * (T_cell - T_jig)
        Q_jig_cool = -self.param.k_jig_air * (T_jig - T_amb)

        kelvin = 273.15
        variables = {
            "Cell temperature [degC]": T_cell,
            "Cell temperature [K]": T_cell + kelvin,
            "Jig temperature [degC]": T_jig,
            "Jig temperature [K]": T_jig + kelvin,
            "Ambient temperature [degC]": T_amb,
            "Ambient temperature [K]": T_amb + kelvin,
            "Heat transfer from cell to jig [W]": Q_cell_cool,
            "Heat transfer from jig to ambient [W]": Q_jig_cool,
        }

        return variables

    def get_coupled_variables(self, variables):
        number_of_rc_elements = self.model_options["number of rc elements"]
        number_of_elements = number_of_rc_elements + 1

        Q_irr = pybamm.Scalar(0)
        for i in range(number_of_elements):
            Q_irr += variables[f"Element-{i} irreversible heat generation [W]"]

        Q_rev = variables["Reversible heat generation [W]"]

        variables.update(
            {
                "Irreversible heat generation [W]": Q_irr,
                "Total heat generation [W]": Q_irr + Q_rev,
            }
        )

        return variables

    def set_rhs(self, variables):
        T_cell = variables["Cell temperature [degC]"]
        T_jig = variables["Jig temperature [degC]"]

        Q_irr = variables["Irreversible heat generation [W]"]
        Q_rev = variables["Reversible heat generation [W]"]

        Q_cell_cool = variables["Heat transfer from cell to jig [W]"]
        Q_jig_cool = variables["Heat transfer from jig to ambient [W]"]

        self.rhs = {
            T_cell: (Q_irr + Q_rev + Q_cell_cool) / self.param.cth_cell,
            T_jig: (Q_jig_cool - Q_cell_cool) / self.param.cth_jig,
        }

    def set_initial_conditions(self, variables):
        T_cell = variables["Cell temperature [degC]"]
        T_jig = variables["Jig temperature [degC]"]
        self.initial_conditions = {
            T_cell: self.param.initial_T_cell,
            T_jig: self.param.initial_T_jig,
        }
