#
# Model for a lithium metal counter-electrode
#
import pybamm


class FullLithiumMetalElectrode(pybamm.BaseSubModel):
    """Class for lithium metal counter-electrode submodel.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        l_Li = pybamm.standard_variables.l_Li

        variables = {
            "Lithium metal electrode thickness": l_Li,
            "Lithium metal electrode thickness [m]": l_Li * self.param.L_x,
        }

        return variables

    def get_coupled_variables(self, variables):
        return variables

    def set_rhs(self, variables):
        l_Li = variables["Lithium metal electrode thickness"]
        self.rhs = {l_Li: -j_Li}

    def set_initial_conditions(self, variables):
        l_Li = variables["Lithium metal electrode thickness"]
        self.initial_conditions = {l_Li: self.param.l_n}
