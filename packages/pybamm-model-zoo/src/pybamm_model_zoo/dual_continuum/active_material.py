"""Constant active material with an image-based specific surface area."""

from __future__ import annotations

import pybamm


class ConstantFromImage(pybamm.active_material.Constant):
    """Constant active material whose specific surface area comes from images.

    The reactive specific surface area is

    .. math::
        a = a_{AM-e} + \\varepsilon^{surf}_{CBD} a_{AM-CBD}
            + \\varepsilon^{surf}_{sep} a_{AM-sep},

    where the AM-CBD and AM-separator areas are weighted by surface porosities,
    since the CBD and the separator are effective media that contain
    electrolyte. An effective particle radius :math:`R_{eff} = 3\\varepsilon_s/a`
    is also provided.
    """

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        variables = super().get_fundamental_variables()

        x = pybamm.SpatialVariable(
            f"x_{domain[0]}",
            domain=[f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="cartesian",
        )
        inputs = {"Through-cell distance (x) [m]": x}
        prefix = f"{self.phase_param.phase_prefactor}{Domain} electrode specific "
        a_electrolyte = pybamm.FunctionParameter(
            f"{prefix}surface area from image (AM-electrolyte) [m-1]", inputs
        )
        a_cbd = pybamm.FunctionParameter(
            f"{prefix}surface area from image (AM-CBD) [m-1]", inputs
        )
        a_separator = pybamm.FunctionParameter(
            f"{prefix}surface area from image (AM-separator) [m-1]", inputs
        )
        porosity_cbd = pybamm.Parameter("CBD surface porosity")
        porosity_separator = pybamm.Parameter("Separator surface porosity")

        a = a_electrolyte + porosity_cbd * a_cbd + porosity_separator * a_separator
        a.print_name = f"a_{domain[0]}"
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        variables.update(
            {
                f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]": a,
                f"X-averaged {domain} electrode {phase_name}"
                "surface area to volume ratio [m-1]": pybamm.x_average(a),
            }
        )
        variables.update(effective_radius_variables(domain, phase_name, eps_s, a))
        return variables


def effective_radius_variables(domain, phase_name, eps_s, a):
    """Effective particle radius R_eff = 3 eps_s / a, consistent with a."""
    Domain = domain.capitalize()
    R_eff = 3 * eps_s / a
    return {
        f"{Domain} electrode {phase_name}effective particle radius [m]": R_eff,
        f"X-averaged {domain} electrode {phase_name}"
        "effective particle radius [m]": pybamm.x_average(R_eff),
    }


class ConstantSpherical(pybamm.active_material.Constant):
    """PyBaMM's constant active material (a = 3 eps_s / R) plus R_eff = R."""

    def get_fundamental_variables(self):
        Domain = self.domain.capitalize()
        phase_name = self.phase_name
        variables = super().get_fundamental_variables()
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        variables.update(effective_radius_variables(self.domain, phase_name, eps_s, a))
        return variables
