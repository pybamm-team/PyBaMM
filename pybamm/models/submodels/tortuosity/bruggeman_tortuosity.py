#
# Class for Bruggeman tortuosity
#
import pybamm

from .base_tortuosity import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman tortuosity

    **Extends:** :class:`pybamm.tortuosity.BaseModel`
    """

    def __init__(self, param, phase, set_leading_order=False):
        super().__init__(param, phase)
        self.set_leading_order = set_leading_order

    def get_coupled_variables(self, variables):
        param = self.param

        if self.phase == "Electrolyte":
            eps_n, eps_s, eps_p = variables["Porosity"].orphans
            tor = pybamm.Concatenation(
                eps_n ** param.b_e_n, eps_s ** param.b_e_s, eps_p ** param.b_e_p
            )
        elif self.phase == "Electrode":
            eps_n = variables["Negative electrode active material volume fraction"]
            tor_s = pybamm.FullBroadcast(0, "separator", "current collector")
            eps_p = variables["Positive electrode active material volume fraction"]
            tor = pybamm.Concatenation(
                eps_n ** param.b_s_n, tor_s, eps_p ** param.b_s_p
            )

        variables.update(
            self._get_standard_tortuosity_variables(tor, self.set_leading_order)
        )

        return variables
