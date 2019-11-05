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

    def __init__(self, param, domain, reactions=None, set_positive_potential=True):
        super().__init__(param, domain, reactions, set_positive_potential)

    def set_boundary_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        eps = variables[self.domain + " electrode porosity"]
        i_boundary_cc = variables["Current collector current density"]
        phi_s_cn = variables["Negative current collector potential"]

        if self.domain == "Negative":
            lbc = (phi_s_cn, "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.sigma_p * (1 - eps) ** self.param.b_p
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        if pybamm.have_idaklu():
            return pybamm.IDAKLUSolver()
        else:
            return pybamm.CasadiSolver()
