#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_diffusion import BaseModel


class ConstantConcentration(BaseModel):
    """Class for constant concentration of electrolyte

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        c_e_n = pybamm.FullBroadcast(1, "negative electrode", "current collector")
        c_e_s = pybamm.FullBroadcast(1, "separator", "current collector")
        c_e_p = pybamm.FullBroadcast(1, "positive electrode", "current collector")
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        variables = self._get_standard_concentration_variables(c_e)

        N_e = pybamm.FullBroadcast(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_boundary_conditions(self, variables):
        return None
