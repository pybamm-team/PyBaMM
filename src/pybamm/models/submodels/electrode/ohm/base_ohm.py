#
# Base class for Ohm's law submodels
#
import pybamm
from pybamm.models.submodels.electrode.base_electrode import BaseElectrode


class BaseModel(BaseElectrode):
    """A base class for electrode submodels that employ
    Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model."""

    def __init__(self, param, domain, options=None, set_positive_potential=True):
        super().__init__(param, domain, options, set_positive_potential)

    def set_boundary_conditions(self, variables):
        Domain = self.domain.capitalize()

        if self.options.electrode_types["negative"] == "planar":
            return

        if self.domain == "negative":
            phi_s_cn = variables["Negative current collector potential [V]"]
            lbc = (phi_s_cn, "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            T_p = variables["Positive electrode temperature [K]"]
            sigma_eff = (
                self.param.p.sigma(T_p)
                * variables["Positive electrode transport efficiency"]
            )
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        phi_s = variables[f"{Domain} electrode potential [V]"]
        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
