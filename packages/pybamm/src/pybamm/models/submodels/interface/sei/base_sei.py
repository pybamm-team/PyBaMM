#
# Base class for SEI models.
#
import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


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

    def _get_standard_concentration_variables(self, c_sei):
        """
        A private function to obtain the standard variables which
        can be derived from the local SEI concentration.

        Parameters
        ----------
        c : :class:`pybamm.Symbol`
            The SEI concentration.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI concentration.
        """
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        reaction_name = self.reaction_name

        if self.reaction_loc == "interface":
            # c_sei is an interfacial quantity [mol.m-2]
            c_sei_av = pybamm.yz_average(c_sei)
            variables = {
                f"{Domain} {reaction_name}concentration [mol.m-2]": c_sei,
                f"Volume-averaged {domain} {reaction_name}concentration "
                "[mol.m-2]": c_sei_av,
            }
        else:
            # c_sei is a bulk quantity [mol.m-3]
            if self.size_distribution:
                c_sei_sav = pybamm.size_average(c_sei)
                c_sei_xav = pybamm.x_average(c_sei_sav)
            else:
                c_sei_xav = pybamm.x_average(c_sei)
            c_sei_av = pybamm.yz_average(c_sei_xav)
            variables = {
                f"{Domain} {self.reaction_name}concentration [mol.m-3]": c_sei,
                f"X-averaged {domain} {reaction_name}concentration "
                "[mol.m-3]": c_sei_xav,
                f"Volume-averaged {domain} {reaction_name}concentration "
                "[mol.m-3]": c_sei_av,
            }

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            c_sei_0 = 0
            c_sei_cr0 = 0
            z_sei = 1
        else:
            L_sei_0 = phase_param.L_sei_0
            V_bar_sei = phase_param.V_bar_sei
            z_sei = phase_param.z_sei
            if self.reaction_loc == "interface":
                c_sei_0 = L_sei_0 / V_bar_sei  # mol.m-2
            else:
                c_sei_0 = L_sei_0 * phase_param.a_typ / V_bar_sei  # mol.m-3
                c_sei_cr0 = phase_param.L_sei_cr0 * phase_param.a_typ / V_bar_sei

        if self.reaction == "SEI":
            delta_c_sei = c_sei_av - c_sei_0
        elif self.reaction == "SEI on cracks":
            delta_c_sei = c_sei_av - c_sei_cr0

        if self.reaction_loc == "interface":
            L_k = 1
        elif domain == "negative":
            L_k = self.param.n.L
        elif domain == "positive":
            L_k = self.param.p.L

        # Multiply delta_n_SEI by V_k to get total moles of SEI formed
        # Multiply by z_sei to get total lithium moles consumed by SEI
        V_k = L_k * self.param.L_y * self.param.L_z
        Q_sei = delta_c_sei * V_k * z_sei

        variables.update(
            {
                f"Loss of lithium to {domain} {self.reaction_name}[mol]": Q_sei,
                f"Loss of capacity to {domain} {self.reaction_name}[A.h]": Q_sei
                * self.param.F
                / 3600,
            }
        )

        return variables

    def _get_standard_reaction_variables(self, j_sei):
        """
        A private function to obtain the standard variables which
        can be derived from the SEI interfacial reaction current

        Parameters
        ----------
        j_sei : :class:`pybamm.Symbol`
            The SEI interfacial reaction current.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI currents.
        """
        domain, Domain = self.domain_Domain

        variables = {
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

    def _get_standard_reaction_distribution_variables(self, j_sei):
        """
        A private function to obtain the standard variables which
        can be derived from the SEI interfacial reaction current

        Parameters
        ----------
        j_sei : :class:`pybamm.Symbol`
            The SEI interfacial reaction current.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI currents.
        """
        domain, Domain = self.domain_Domain
        j_sei = pybamm.size_average(j_sei)
        variables = {
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
