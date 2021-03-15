#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class ConstantConcentration(BaseElectrolyteDiffusion):
    """Class for constant concentration of electrolyte

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        c_e_n = pybamm.FullBroadcast(1, "negative electrode", "current collector")
        c_e_s = pybamm.FullBroadcast(1, "separator", "current collector")
        c_e_p = pybamm.FullBroadcast(1, "positive electrode", "current collector")

        variables = self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)

        N_e = pybamm.FullBroadcastToEdges(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def get_coupled_variables(self, variables):
        c_e = variables["Electrolyte concentration"]
        eps = variables["Porosity"]

        variables.update(self._get_total_concentration_electrolyte(c_e, eps))

        return variables

    def set_boundary_conditions(self, variables):
        """
        We provide boundary conditions even though the concentration is constant
        so that the gradient of the concentration has the correct shape after
        discretisation.
        """

        c_e = variables["Electrolyte concentration"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
