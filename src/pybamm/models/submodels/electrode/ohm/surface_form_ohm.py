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

    def build(self, submodels):
        Domain = self.domain.capitalize()

        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        i_e = pybamm.CoupledVariable(
            f"{Domain} electrolyte current density [A.m-2]",
            f"{self.domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({i_e.name: i_e})
        tor = pybamm.CoupledVariable(
            f"{Domain} electrode transport efficiency",
            f"{self.domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor.name: tor})
        phi_s_cn = pybamm.CoupledVariable(
            "Negative current collector potential [V]",
            domain="current collector",
        )
        self.coupled_variables.update({phi_s_cn.name: phi_s_cn})
        T = pybamm.CoupledVariable(
            f"{Domain} electrode temperature [K]",
            f"{self.domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({T.name: T})

        conductivity = self.domain_param.sigma(T) * tor
        i_s = i_boundary_cc - i_e

        if self.domain == "negative":
            phi_s = phi_s_cn - pybamm.IndefiniteIntegral(i_s / conductivity, x_n)

        elif self.domain == "positive":
            phi_e_s = pybamm.CoupledVariable(
                "Separator electrolyte potential [V]",
                domain="separator",
            )
            self.coupled_variables.update({phi_e_s.name: phi_e_s})
            delta_phi_p = pybamm.CoupledVariable(
                "Positive electrode surface potential difference [V]",
                domain="current collector",
            )
            self.coupled_variables.update({delta_phi_p.name: delta_phi_p})
            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_p) + (
                pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        variables = self._get_standard_potential_variables(phi_s)
        variables.update(self._get_standard_current_variables(i_s))

        if (
            self.options.electrode_types["negative"] == "planar"
            or "Negative electrode current density [A.m-2]" in variables
        ) and "Positive electrode current density [A.m-2]" in variables:
            variables.update(self._get_standard_whole_cell_variables(variables))

        self.variables.update(variables)
