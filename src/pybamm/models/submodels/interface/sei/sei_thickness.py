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

    def __init__(self, param, domain, options, phase="primary", cracks=False):
        super().__init__(param, domain, options=options, phase=phase, cracks=cracks)
        if self.options.electrode_types[domain] == "planar":
            self.reaction_loc = "interface"
        elif self.options["x-average side reactions"] == "true":
            self.reaction_loc = "x-average"
        else:
            self.reaction_loc = "full electrode"

    def get_coupled_variables(self, variables):
        """Update variables related to the SEI thickness."""
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        reaction_name = self.reaction_name
        SEI_option = getattr(self.options, domain)["SEI"]
        crack_option = getattr(self.options, domain)["SEI on cracks"]
        if self.options["working electrode"] != "both" and domain == "negative":
            crack_option = "false"  # required if SEI on cracks is used for half-cells

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if SEI_option == "none":
            c_to_L = 1
            R_sei = 1
        else:
            if self.reaction_loc == "interface":
                c_to_L = phase_param.V_bar_sei
            else:
                a = variables[
                    f"{Domain} electrode {self.phase_name}"
                    "surface area to volume ratio [m-1]"
                ]
                c_to_L = phase_param.V_bar_sei / a
            R_sei = phase_param.R_sei

        if self.reaction_loc == "interface":
            # c_sei is an interfacial quantity [mol.m-2]
            c_sei = variables[f"{Domain} {reaction_name}concentration [mol.m-2]"]
        else:
            # c_sei is a bulk quantity [mol.m-3]
            c_sei = variables[f"{Domain} {reaction_name}concentration [mol.m-3]"]

        if self.reaction == "SEI":
            L_sei = c_sei * c_to_L  # SEI thickness
            if self.size_distribution:
                L_sei_sav = pybamm.size_average(L_sei)  # size-averaged SEI thickness
                L_sei_xav = pybamm.x_average(L_sei_sav)  # x-averaged SEI thickness
            else:
                L_sei_xav = pybamm.x_average(L_sei)  # x-averaged SEI thickness
            L_sei_av = pybamm.yz_average(L_sei_xav)  # volume-averaged SEI thickness

            variables.update(
                {
                    f"X-averaged {self.domain} electrode resistance [Ohm.m2]": L_sei_xav
                    * R_sei,
                }
            )
        # Thickness variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            # if SEI on cracks is false, skip over roughness to avoid division by zero
            if crack_option == "false":
                L_sei = c_sei * c_to_L
            else:
                roughness = variables[f"{Domain} electrode roughness ratio"]
                L_sei = c_sei * c_to_L / (roughness - 1)  # SEI on cracks thickness
            if self.size_distribution:
                L_sei_sav = pybamm.size_average(L_sei)
                L_sei_xav = pybamm.x_average(L_sei_sav)
            else:
                L_sei_xav = pybamm.x_average(L_sei)  # x-average SEI on cracks thickness
            L_sei_av = pybamm.yz_average(L_sei_xav)  # average SEI on cracks thickness

        variables.update(
            {
                f"{Domain} {reaction_name}thickness [m]": L_sei,
                f"X-averaged {domain} {reaction_name}thickness [m]": L_sei_xav,
                f"Volume-averaged {domain} {reaction_name}thickness [m]": L_sei_av,
            }
        )

        return variables
