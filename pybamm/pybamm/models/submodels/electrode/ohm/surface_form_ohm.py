#
# Class for ohmic electrodes in the surface potential formulation
#
import pybamm

from .base_ohm import BaseModel


class SurfaceForm(BaseModel):
    """A submodel for the electrode with Ohm's law in the surface potential
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()

        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        i_e = variables[f"{Domain} electrolyte current density [A.m-2]"]
        tor = variables[f"{Domain} electrode transport efficiency"]
        phi_s_cn = variables["Negative current collector potential [V]"]
        T = variables[f"{Domain} electrode temperature [K]"]

        conductivity = self.domain_param.sigma(T) * tor
        i_s = i_boundary_cc - i_e

        if self.domain == "negative":
            phi_s = phi_s_cn - pybamm.IndefiniteIntegral(i_s / conductivity, x_n)

        elif self.domain == "positive":
            phi_e_s = variables["Separator electrolyte potential [V]"]
            delta_phi_p = variables[
                "Positive electrode surface potential difference [V]"
            ]

            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_p) + (
                pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        variables.update(self._get_standard_potential_variables(phi_s))
        variables.update(self._get_standard_current_variables(i_s))

        if (
            self.options.electrode_types["negative"] == "planar"
            or "Negative electrode current density [A.m-2]" in variables
        ) and "Positive electrode current density [A.m-2]" in variables:
            variables.update(self._get_standard_whole_cell_variables(variables))

        return variables
