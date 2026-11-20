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

    def build(self):
        Domain = self.domain.capitalize()
        coupled = self._coupled_electrode_variable

        i_boundary_cc = self._coupled_current_collector_variable(
            "Current collector current density [A.m-2]"
        )
        i_e = coupled(f"{Domain} electrolyte current density [A.m-2]")
        # A placeholder cannot know that the electrolyte current is evaluated on
        # edges, which decides whether its indefinite integral is
        i_e._saved_evaluates_on_edges = {"primary": True}
        conductivity = self._get_effective_conductivity()
        i_s = i_boundary_cc - i_e

        if self.domain == "negative":
            x_n = pybamm.standard_spatial_vars.x_n
            phi_s_cn = self._coupled_current_collector_variable(
                "Negative current collector potential [V]"
            )
            phi_s = phi_s_cn - pybamm.IndefiniteIntegral(i_s / conductivity, x_n)
        else:
            x_p = pybamm.standard_spatial_vars.x_p
            phi_e_s = self.coupled_variable(
                "Separator electrolyte potential [V]",
                domains={"primary": "separator", "secondary": "current collector"},
            )
            delta_phi_p = coupled("Positive electrode surface potential difference [V]")
            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_p) + (
                pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        self.variables.update(self._get_standard_potential_variables(phi_s))
        self.variables.update(self._get_standard_current_variables(i_s))
        if self.domain == "positive":
            self.variables.update(self._get_standard_whole_cell_variables(i_s, phi_s))

        self._set_porous_boundary_conditions(phi_s, conductivity)
