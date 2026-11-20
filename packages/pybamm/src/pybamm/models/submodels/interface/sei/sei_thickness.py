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

    def build(self):
        """Set variables related to the SEI thickness."""
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
                a = self._coupled_electrode_variable(
                    f"electrode {self.phase_name}surface area to volume ratio [m-1]"
                )
                c_to_L = phase_param.V_bar_sei / a
            R_sei = phase_param.R_sei

        c_sei = self._get_coupled_sei_concentration()

        if self.reaction == "SEI":
            L_sei = c_sei * c_to_L  # SEI thickness
            if self.size_distribution:
                L_sei_sav = pybamm.size_average(L_sei)  # size-averaged SEI thickness
                L_sei_xav = pybamm.x_average(L_sei_sav)  # x-averaged SEI thickness
            else:
                L_sei_xav = pybamm.x_average(L_sei)  # x-averaged SEI thickness
            L_sei_av = pybamm.yz_average(L_sei_xav)  # volume-averaged SEI thickness

            self.variables[
                f"X-averaged {self.domain} electrode resistance [Ohm.m2]"
            ] = L_sei_xav * R_sei
        # Thickness variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            # if SEI on cracks is false, skip over roughness to avoid division by zero
            if crack_option == "false":
                L_sei = c_sei * c_to_L
            else:
                roughness = self._coupled_electrode_variable(
                    "electrode roughness ratio"
                )
                L_sei = c_sei * c_to_L / (roughness - 1)  # SEI on cracks thickness
            if self.size_distribution:
                L_sei_sav = pybamm.size_average(L_sei)
                L_sei_xav = pybamm.x_average(L_sei_sav)
            else:
                L_sei_xav = pybamm.x_average(L_sei)  # x-average SEI on cracks thickness
            L_sei_av = pybamm.yz_average(L_sei_xav)  # average SEI on cracks thickness

        self.variables.update(
            {
                f"{Domain} {reaction_name}thickness [m]": L_sei,
                f"X-averaged {domain} {reaction_name}thickness [m]": L_sei_xav,
                f"Volume-averaged {domain} {reaction_name}thickness [m]": L_sei_av,
            }
        )

    def _get_coupled_sei_concentration(self):
        """
        Placeholder for the concentration set by the SEI submodel of this
        reaction, on the domains that submodel gives it.
        """
        domain, Domain = self.domain_Domain
        name = f"{Domain} {self.reaction_name}concentration"
        sei_option = getattr(getattr(self.options, domain), self.phase)["SEI"]
        if self.reaction == "SEI on cracks":
            no_sei = (
                getattr(self.options, domain)["SEI"] in ["none", "constant"]
                or getattr(self.options, domain)["SEI on cracks"] == "false"
            )
            constant = False
        else:
            no_sei = sei_option == "none"
            constant = sei_option == "constant"

        if constant:
            # A constant SEI concentration is a scalar
            unit = "[mol.m-2]" if self.reaction_loc == "interface" else "[mol.m-3]"
            return self.coupled_variable(f"{name} {unit}", domains={})
        if self.reaction_loc == "interface":
            return self.coupled_variable(
                f"{name} [mol.m-2]", domain="current collector"
            )
        return self._coupled_electrode_variable(
            f"{self.reaction_name}concentration [mol.m-3]",
            size=no_sei and self.size_distribution,
        )
