#
# Base class for SEI models.
#
import pybamm
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """Base class for SEI models.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    cracks : bool, optional
        Whether this is a submodel for standard SEI or SEI on cracks
    """

    def __init__(self, param, domain, options, phase="primary", cracks=False):
        if cracks is True:
            reaction = "SEI on cracks"
        else:
            reaction = "SEI"
        super().__init__(param, domain, reaction, options=options, phase=phase)

    def get_coupled_variables(self, variables):
        # Update some common variables
        domain, Domain = self.domain_Domain

        if self.reaction_loc != "interface":
            j_sei_av = variables[
                f"X-averaged {domain} electrode {self.reaction_name}interfacial"
                " current density [A.m-2]"
            ]
            j_sei = variables[
                f"{Domain} electrode {self.reaction_name}interfacial current"
                " density [A.m-2]"
            ]
            variables.update(
                {
                    f"X-averaged {domain} electrode {self.reaction_name}interfacial "
                    "current density [A.m-2]": j_sei_av,
                    f"{Domain} electrode {self.reaction_name}interfacial current "
                    "density [A.m-2]": j_sei,
                }
            )

        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        return variables

    def _get_standard_concentration_variables(self, c_inner, c_outer):
        """
        A private function to obtain the standard variables which
        can be derived from the local SEI concentration.

        Parameters
        ----------
        c_inner : :class:`pybamm.Symbol`
            The inner SEI concentration.
        c_outer : :class:`pybamm.Symbol`
            The outer SEI concentration.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI thicknesses.
        """
        domain, Domain = self.domain_Domain
        if self.reaction_loc == "interface":  # c is an interfacial quantity [mol.m-2]
            variables = {
                f"{Domain} inner {self.reaction_name}concentration [mol.m-2]": c_inner,
                f"{Domain} outer {self.reaction_name}concentration [mol.m-2]": c_outer,
            }
        else:  # c is a bulk quantity [mol.m-3]
            c_inner_av = pybamm.x_average(c_inner)
            c_outer_av = pybamm.x_average(c_outer)
            variables = {
                f"{Domain} inner {self.reaction_name}concentration [mol.m-3]": c_inner,
                f"{Domain} outer {self.reaction_name}concentration [mol.m-3]": c_outer,
                f"X-averaged {domain} inner {self.reaction_name}"
                "concentration [mol.m-3]": c_inner_av,
                f"X-averaged {domain} outer {self.reaction_name}"
                "concentration [mol.m-3]": c_outer_av,
            }

        # Get variables related to the total concentration
        c_sei = c_inner + c_outer
        variables.update(self._get_standard_total_concentration_variables(c_sei))

        return variables

    def _get_standard_total_concentration_variables(self, c_sei):
        """Update variables related to total SEI concentration."""
        domain, Domain = self.domain_Domain
        if self.reaction_loc == "interface":  # c is an interfacial quantity [mol.m-2]
            variables = {
                f"{Domain} {self.reaction_name}concentration [mol.m-2]": c_sei,
                f"{Domain} total {self.reaction_name}concentration [mol.m-2]": c_sei,
            }
        else:  # c is a bulk quantity [mol.m-3]
            c_xav = pybamm.x_average(c_sei)
            variables = {
                f"{Domain} {self.reaction_name}concentration [mol.m-3]": c_sei,
                f"{Domain} total {self.reaction_name}concentration [mol.m-3]": c_sei,
                f"X-averaged {domain} {self.reaction_name}"
                "concentration [mol.m-3]": c_xav,
                f"X-averaged {domain} total {self.reaction_name}"
                "concentration [mol.m-3]": c_xav,
            }
        return variables

    def _get_standard_reaction_variables(self, j_inner, j_outer):
        """
        A private function to obtain the standard variables which
        can be derived from the SEI interfacial reaction current

        Parameters
        ----------
        j_inner : :class:`pybamm.Symbol`
            The inner SEI interfacial reaction current.
        j_outer : :class:`pybamm.Symbol`
            The outer SEI interfacial reaction current.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI currents.
        """
        domain, Domain = self.domain_Domain
        j_inner_av = pybamm.x_average(j_inner)
        j_outer_av = pybamm.x_average(j_outer)
        j_sei = j_inner + j_outer

        variables = {
            f"{Domain} electrode inner {self.reaction_name}"
            "interfacial current density [A.m-2]": j_inner,
            f"X-averaged {domain} electrode inner {self.reaction_name}"
            "interfacial current density [A.m-2]": j_inner_av,
            f"{Domain} electrode outer {self.reaction_name}"
            "interfacial current density [A.m-2]": j_outer,
            f"X-averaged {domain} electrode outer {self.reaction_name}"
            "interfacial current density [A.m-2]": j_outer_av,
            f"{Domain} electrode {self.reaction_name}"
            "interfacial current density [A.m-2]": j_sei,
        }

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {domain} electrode {self.reaction_name}"
                    "interfacial current density [A.m-2]": j_sei_av,
                }
            )

        return variables
