#
# Class for Bruggeman tortuosity
#
import pybamm

from .base_tortuosity import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman tortuosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    phase : str
        The material for the model ('electrolyte' or 'electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.tortuosity.BaseModel`
    """

    def __init__(self, param, phase, options=None, set_leading_order=False):
        super().__init__(param, phase, options=options)
        self.set_leading_order = set_leading_order

    def get_coupled_variables(self, variables):
        param = self.param

        if self.phase == "Electrolyte":
            if self.half_cell:
                tor_n = None
            else:
                eps_n = variables["Negative electrode porosity"]
                tor_n = eps_n ** param.b_e_n

            eps_s = variables["Separator porosity"]
            tor_s = eps_s ** param.b_e_s
            eps_p = variables["Positive electrode porosity"]
            tor_p = eps_p ** param.b_e_p
        elif self.phase == "Electrode":
            if self.half_cell:
                tor_n = None
            else:
                eps_n = variables["Negative electrode active material volume fraction"]
                tor_n = eps_n ** param.b_s_n

            eps_p = variables["Positive electrode active material volume fraction"]
            tor_s = pybamm.FullBroadcast(0, "separator", "current collector")
            tor_p = eps_p ** param.b_s_p

        variables.update(
            self._get_standard_tortuosity_variables(
                tor_n, tor_s, tor_p, self.set_leading_order
            )
        )

        return variables
