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

    def _get_standard_thickness_variables(self, L):
        """
        A private function to obtain the standard variables which
        can be derived from the local SEI thickness.

        Parameters
        ----------
        L : :class:`pybamm.Symbol`
            The SEI thickness.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI thicknesses.
        """
        domain, Domain = self.domain_Domain
        variables = {f"{Domain} {self.reaction_name}thickness [m]": L}

        if self.reaction_loc != "interface":
            L_av = pybamm.x_average(L)
            variables.update(
                {
                    f"X-averaged {domain} {self.reaction_name}thickness [m]": L_av,
                }
            )
        # Get variables related to the total thickness
        variables.update(self._get_standard_total_thickness_variables(L))

        return variables

    def _get_standard_total_thickness_variables(self, L_sei):
        """Update variables related to total SEI thickness."""
        domain, Domain = self.domain_Domain

        if isinstance(self, pybamm.sei.NoSEI):
            R_sei = 1
        else:
            R_sei = self.phase_param.R_sei

        variables = {
            f"{Domain} {self.reaction_name}[m]": L_sei,
            f"{Domain} total {self.reaction_name}thickness [m]": L_sei,
        }
        if self.reaction_loc != "interface":
            L_sei_av = pybamm.x_average(L_sei)
            variables.update(
                {
                    f"X-averaged {domain} {self.reaction_name}thickness [m]": L_sei_av,
                    f"X-averaged {domain} total {self.reaction_name}"
                    "thickness [m]": L_sei_av,
                }
            )
            if self.reaction == "SEI":
                variables.update(
                    {
                        f"X-averaged {domain} electrode resistance [Ohm.m2]": L_sei_av
                        * R_sei,
                    }
                )
        return variables

    def _get_standard_concentration_variables(self, variables):
        """Update variables related to the SEI concentration."""
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        phase_name = phase_param.phase_name
        reaction_name = self.reaction_name

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_to_n = 0
            n_SEI_0 = 0
            n_crack_0 = 0
            z_sei = 1
        else:
            if self.reaction_loc == "interface":
                # m * (mol/m3) = mol/m2 (n is an interfacial quantity)
                L_to_n = 1 / phase_param.V_bar_sei
                L_to_n_0 = L_to_n
            else:
                # m * (mol/m4) = mol/m3 (n is a bulk quantity)
                a = variables[
                    f"{Domain} electrode {self.phase_name}"
                    "surface area to volume ratio [m-1]"
                ]
                L_to_n = a / phase_param.V_bar_sei
                L_to_n_0 = phase_param.a_typ / phase_param.V_bar_sei
            z_sei = phase_param.z_sei
            L_0 = phase_param.L_sei_0
            L_crack_0 = phase_param.L_sei_crack_0
            n_SEI_0 = L_0 * L_to_n_0
            n_crack_0 = L_crack_0 * L_to_n_0

        if self.reaction == "SEI":
            L = variables[f"{Domain} {reaction_name}thickness [m]"]

            n_SEI = L * L_to_n
            if self.size_distribution:
                n_SEI_sav = pybamm.size_average(n_SEI)
                n_SEI_xav = pybamm.x_average(n_SEI_sav)
            else:
                n_SEI_xav = pybamm.x_average(n_SEI)
            n_SEI_av = pybamm.yz_average(n_SEI_xav)

            # Calculate change in SEI concentration with respect to initial state
            delta_n_SEI = n_SEI_av - n_SEI_0

            # Q_sei in mol
            if self.reaction_loc == "interface":
                L_k = 1
            elif domain == "negative":
                L_k = self.param.n.L
            elif domain == "positive":
                L_k = self.param.p.L

            # Multiply delta_n_SEI by V_k to get total moles of SEI formed
            # multiply by z_sei to get total lithium moles consumed by SEI
            V_k = L_k * self.param.L_y * self.param.L_z
            Q_sei = z_sei * delta_n_SEI * V_k

            variables.update(
                {
                    f"{Domain} {reaction_name}concentration [mol.m-3]": n_SEI,
                    f"X-averaged {domain} {reaction_name}"
                    "concentration [mol.m-3]": n_SEI_xav,
                    f"Loss of lithium to {domain} {reaction_name}[mol]": Q_sei,
                    f"Loss of capacity to {domain} {reaction_name}[A.h]": Q_sei
                    * self.param.F
                    / 3600,
                }
            )
        # Concentration variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            L_cr = variables[f"{Domain} {reaction_name}thickness [m]"]
            roughness = variables[f"{Domain} {phase_name}electrode roughness ratio"]

            n_SEI_cr = L_cr * L_to_n * (roughness - 1)  # SEI on cracks concentration
            if self.size_distribution:
                n_SEI_cr_sav = pybamm.size_average(n_SEI_cr)
                n_SEI_cr_xav = pybamm.x_average(n_SEI_cr_sav)
                n_SEI_cr_av = pybamm.yz_average(n_SEI_cr_xav)
            else:
                n_SEI_cr_xav = pybamm.x_average(n_SEI_cr)
                n_SEI_cr_av = pybamm.yz_average(n_SEI_cr_xav)

            # Calculate change in SEI cracks concentration
            # Initial state depends on roughness (to avoid division by zero)
            roughness_av = pybamm.yz_average(pybamm.x_average(roughness))
            # choose an initial condition that is as close to zero to get the
            # physics right, but doesn't cause a division by zero error
            n_SEI_cr_init = n_crack_0 * (roughness_av - 1)
            delta_n_SEI_cr = n_SEI_cr_av - n_SEI_cr_init

            if domain == "negative":
                L_k = self.param.n.L
            elif domain == "positive":
                L_k = self.param.p.L

            # Q_sei_cr in mol
            Q_sei_cr = z_sei * delta_n_SEI_cr * L_k * self.param.L_y * self.param.L_z

            variables.update(
                {
                    f"{Domain} {reaction_name}concentration [mol.m-3]": n_SEI_cr,
                    f"X-averaged {domain} {reaction_name}"
                    "concentration [mol.m-3]": n_SEI_cr_xav,
                    f"Loss of lithium to {domain} {reaction_name}[mol]": Q_sei_cr,
                    f"Loss of capacity to {domain} {reaction_name}[A.h]": Q_sei_cr
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
