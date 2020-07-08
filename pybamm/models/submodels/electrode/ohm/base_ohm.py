#
# Base class for Ohm's law submodels
#
import pybamm
from ..base_electrode import BaseElectrode


class BaseModel(BaseElectrode):
    """A base class for electrode submodels that employ
    Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.electrode.BaseElectrode`
    """

    def __init__(self, param, domain, set_positive_potential=True):
        super().__init__(param, domain, set_positive_potential)

    def set_boundary_conditions(self, variables):

        if self.domain == "Negative":
            phi_s_cn = variables["Negative current collector potential"]
            lbc = (phi_s_cn, "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            i_boundary_cc = variables["Current collector current density"]
            sigma_eff = self.param.sigma_p * variables["Positive electrode tortuosity"]
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        phi_s = variables[self.domain + " electrode potential"]
        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
