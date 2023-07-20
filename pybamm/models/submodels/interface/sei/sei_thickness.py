#
# Class for converting SEI concentration into thickness
#
import pybamm
from .base_sei import BaseModel


class SEIThickness(BaseModel):
    """
    Class for converting SEI concentration into thickness

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    cracks : bool, optional
        Whether this is a submodel for standard SEI or SEI on cracks
    """

    def __init__(self, param, reaction_loc, options, phase, cracks=False):
        super().__init__(param, options=options, phase=phase, cracks=cracks)
        self.reaction_loc = reaction_loc

    def get_coupled_variables(self, variables):
        """Update variables related to the SEI thickness."""
        Domain = self.domain.capitalize()
        phase_param = self.phase_param
        reaction_name = self.reaction_name

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if self.options["SEI"] == "none":
            c_to_L_inner = 1
            c_to_L_outer = 1
            R_sei = 1
        else:
            if self.reaction_loc == "interface":
                c_to_L_inner = phase_param.V_bar_inner
                c_to_L_outer = phase_param.V_bar_outer
            else:
                a = variables[
                    f"Negative electrode {self.phase_name}"
                    "surface area to volume ratio [m-1]"
                ]
                c_to_L_inner = phase_param.V_bar_inner / a
                c_to_L_outer = phase_param.V_bar_outer / a
            R_sei = phase_param.R_sei

        if self.reaction_loc == "interface":
            # c is an interfacial quantity [mol.m-2]
            c_inner = variables[f"Inner {reaction_name}concentration [mol.m-2]"]
            c_outer = variables[f"Outer {reaction_name}concentration [mol.m-2]"]
        else:
            # c is a bulk quantity [mol.m-3]
            c_inner = variables[f"Inner {reaction_name}concentration [mol.m-3]"]
            c_outer = variables[f"Outer {reaction_name}concentration [mol.m-3]"]

        if self.reaction == "SEI":
            L_inner = c_inner * c_to_L_inner  # inner SEI thickness
            L_outer = c_outer * c_to_L_outer  # outer SEI thickness

            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)

            L_SEI = L_inner + L_outer  # SEI thickness
            L_SEI_av = pybamm.x_average(L_SEI)

            variables.update(
                {
                    f"X-averaged {self.domain} electrode resistance "
                    "[Ohm.m2]": L_SEI_av * R_sei,
                }
            )
        # Thickness variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            # if SEI on cracks is false, skip over roughness to avoid division by zero
            if self.options["SEI on cracks"] == "false":
                L_inner = c_inner * c_to_L_inner
                L_outer = c_outer * c_to_L_outer
            else:
                roughness = variables[f"{Domain} electrode roughness ratio"]
                L_inner = c_inner * c_to_L_inner / (roughness - 1)
                L_outer = c_outer * c_to_L_outer / (roughness - 1)

            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)

            L_SEI = L_inner + L_outer
            L_SEI_av = pybamm.x_average(L_SEI)

        variables.update(
            {
                f"Inner {reaction_name}thickness [m]": L_inner,
                f"X-averaged inner {reaction_name}thickness [m]": L_inner_av,
                f"Outer {reaction_name}thickness [m]": L_outer,
                f"X-averaged outer {reaction_name}thickness [m]": L_outer_av,
                f"{reaction_name}thickness [m]": L_SEI,
                f"X-averaged {reaction_name}thickness [m]": L_SEI_av,
                f"Total {reaction_name}thickness [m]": L_SEI,
                f"X-averaged total {reaction_name}thickness [m]": L_SEI_av,
            }
        )
        return variables
