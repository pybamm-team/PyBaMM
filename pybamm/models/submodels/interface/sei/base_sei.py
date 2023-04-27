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

    def __init__(self, param, options, phase="primary", cracks=False):
        if cracks is True:
            reaction = "SEI on cracks"
        else:
            reaction = "SEI"
        domain = "negative"
        super().__init__(param, domain, reaction, options=options, phase=phase)

    def get_coupled_variables(self, variables):
        # Update some common variables

        if self.reaction_loc != "interface":
            j_sei_av = variables[
                f"X-averaged {self.reaction_name}interfacial current density [A.m-2]"
            ]
            j_sei = variables[
                f"{self.reaction_name}interfacial current density [A.m-2]"
            ]
            variables.update(
                {
                    f"X-averaged negative electrode {self.reaction_name}interfacial "
                    "current density [A.m-2]": j_sei_av,
                    f"Negative electrode {self.reaction_name}interfacial current "
                    "density [A.m-2]": j_sei,
                }
            )

        zero_av = pybamm.PrimaryBroadcast(0, "current collector")
        zero = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        variables.update(
            {
                f"Positive electrode {self.reaction} "
                "interfacial current density [A.m-2]": zero,
                f"X-averaged positive electrode {self.reaction} "
                "volumetric interfacial current density [A.m-2]": zero_av,
                f"Positive electrode {self.reaction} "
                "volumetric interfacial current density [A.m-3]": zero,
            }
        )

        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        return variables

    def _get_standard_thickness_variables(self, L_inner, L_outer):
        """
        A private function to obtain the standard variables which
        can be derived from the local SEI thickness.

        Parameters
        ----------
        L_inner : :class:`pybamm.Symbol`
            The inner SEI thickness.
        L_outer : :class:`pybamm.Symbol`
            The outer SEI thickness.

        Returns
        -------
        variables : dict
            The variables which can be derived from the SEI thicknesses.
        """
        variables = {
            f"Inner {self.reaction_name}thickness [m]": L_inner,
            f"Outer {self.reaction_name}thickness [m]": L_outer,
        }

        if self.reaction_loc != "interface":
            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)
            variables.update(
                {
                    f"X-averaged inner {self.reaction_name}thickness [m]": L_inner_av,
                    f"X-averaged outer {self.reaction_name}thickness [m]": L_outer_av,
                }
            )
        # Get variables related to the total thickness
        L_sei = L_inner + L_outer
        variables.update(self._get_standard_total_thickness_variables(L_sei))

        return variables

    def _get_standard_total_thickness_variables(self, L_sei):
        """Update variables related to total SEI thickness."""
        domain = self.domain

        if isinstance(self, pybamm.sei.NoSEI):
            R_sei = 1
        else:
            R_sei = self.phase_param.R_sei

        variables = {
            f"{self.reaction_name}[m]": L_sei,
            f"Total {self.reaction_name}thickness [m]": L_sei,
        }
        if self.reaction_loc != "interface":
            L_sei_av = pybamm.x_average(L_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction_name}thickness [m]": L_sei_av,
                    f"X-averaged total {self.reaction_name}thickness [m]": L_sei_av,
                }
            )
            if self.reaction == "SEI":
                variables.update(
                    {
                        f"X-averaged {domain} electrode resistance "
                        "[Ohm.m2]": L_sei_av * R_sei,
                    }
                )
        return variables

    def _get_standard_concentration_variables(self, variables):
        """Update variables related to the SEI concentration."""
        Domain = self.domain.capitalize()
        phase_param = self.phase_param
        reaction_name = self.reaction_name

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_to_n_inner = 0
            L_to_n_outer = 0
            n_SEI_0 = 0
            n_crack_0 = 0
            z_sei = 1
        else:
            if self.reaction_loc == "interface":
                # m * (mol/m3) = mol/m2 (n is an interfacial quantity)
                L_to_n_inner = 1 / phase_param.V_bar_inner
                L_to_n_outer = 1 / phase_param.V_bar_outer
                L_to_n_inner_0 = L_to_n_inner
                L_to_n_outer_0 = L_to_n_outer
            else:
                # m * (mol/m4) = mol/m3 (n is a bulk quantity)
                a = variables[
                    f"Negative electrode {self.phase_name}"
                    "surface area to volume ratio [m-1]"
                ]
                L_to_n_inner = a / phase_param.V_bar_inner
                L_to_n_outer = a / phase_param.V_bar_outer
                L_to_n_inner_0 = phase_param.a_typ / phase_param.V_bar_inner
                L_to_n_outer_0 = phase_param.a_typ / phase_param.V_bar_outer
            z_sei = phase_param.z_sei
            L_inner_0 = phase_param.L_inner_0
            L_outer_0 = phase_param.L_outer_0
            L_inner_crack_0 = phase_param.L_inner_crack_0
            L_outer_crack_0 = phase_param.L_outer_crack_0
            n_SEI_0 = L_inner_0 * L_to_n_inner_0 + L_outer_0 * L_to_n_outer_0
            n_crack_0 = (
                L_inner_crack_0 * L_to_n_inner_0 + L_outer_crack_0 * L_to_n_outer_0
            )

        if self.reaction == "SEI":
            L_inner = variables[f"Inner {reaction_name}thickness [m]"]
            L_outer = variables[f"Outer {reaction_name}thickness [m]"]

            n_inner = L_inner * L_to_n_inner  # inner SEI concentration
            n_outer = L_outer * L_to_n_outer  # outer SEI concentration

            n_inner_av = pybamm.x_average(n_inner)
            n_outer_av = pybamm.x_average(n_outer)

            n_SEI = n_inner + n_outer  # SEI concentration
            n_SEI_xav = pybamm.x_average(n_SEI)
            n_SEI_av = pybamm.yz_average(n_SEI_xav)

            # Calculate change in SEI concentration with respect to initial state
            delta_n_SEI = n_SEI_av - n_SEI_0

            # Q_sei in mol
            if self.reaction_loc == "interface":
                L_n = 1
            else:
                L_n = self.param.n.L

            # Multiply delta_n_SEI by V_n to get total moles of SEI formed
            # multiply by z_sei to get total lithium moles consumed by SEI
            V_n = L_n * self.param.L_y * self.param.L_z
            Q_sei = z_sei * delta_n_SEI * V_n

            variables.update(
                {
                    f"Inner {reaction_name}concentration [mol.m-3]": n_inner,
                    f"X-averaged inner {reaction_name}"
                    "concentration [mol.m-3]": n_inner_av,
                    f"Outer {reaction_name}concentration [mol.m-3]": n_outer,
                    f"X-averaged outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer_av,
                    f"{reaction_name}concentration [mol.m-3]": n_SEI,
                    f"X-averaged {reaction_name}concentration [mol.m-3]": n_SEI_xav,
                    f"Loss of lithium to {reaction_name}[mol]": Q_sei,
                    f"Loss of capacity to {reaction_name}[A.h]": Q_sei
                    * self.param.F
                    / 3600,
                }
            )
        # Concentration variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            L_inner_cr = variables[f"Inner {reaction_name}thickness [m]"]
            L_outer_cr = variables[f"Outer {reaction_name}thickness [m]"]
            roughness = variables[f"{Domain} electrode roughness ratio"]

            n_inner_cr = L_inner_cr * L_to_n_inner * (roughness - 1)
            n_outer_cr = L_outer_cr * L_to_n_outer * (roughness - 1)

            n_inner_cr_av = pybamm.x_average(n_inner_cr)
            n_outer_cr_av = pybamm.x_average(n_outer_cr)

            n_SEI_cr = n_inner_cr + n_outer_cr  # SEI on cracks concentration
            n_SEI_cr_xav = pybamm.x_average(n_SEI_cr)
            n_SEI_cr_av = pybamm.yz_average(n_SEI_cr_xav)

            # Calculate change in SEI cracks concentration
            # Initial state depends on roughness (to avoid division by zero)
            roughness_av = pybamm.yz_average(pybamm.x_average(roughness))
            # choose an initial condition that is as close to zero to get the
            # physics right, but doesn't cause a division by zero error
            n_SEI_cr_init = n_crack_0 * (roughness_av - 1)
            delta_n_SEI_cr = n_SEI_cr_av - n_SEI_cr_init

            # Q_sei_cr in mol
            Q_sei_cr = (
                z_sei
                * delta_n_SEI_cr
                * self.param.n.L
                * self.param.L_y
                * self.param.L_z
            )

            variables.update(
                {
                    f"Inner {reaction_name}" "concentration [mol.m-3]": n_inner_cr,
                    f"X-averaged inner {reaction_name}"
                    "concentration [mol.m-3]": n_inner_cr_av,
                    f"Outer {reaction_name}concentration [mol.m-3]": n_outer_cr,
                    f"X-averaged outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer_cr_av,
                    f"{reaction_name}" "concentration [mol.m-3]": n_SEI_cr,
                    f"X-averaged {reaction_name}"
                    "concentration [mol.m-3]": n_SEI_cr_xav,
                    f"Loss of lithium to {reaction_name}[mol]": Q_sei_cr,
                    f"Loss of capacity to {reaction_name}[A.h]": Q_sei_cr
                    * self.param.F
                    / 3600,
                }
            )

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
        j_inner_av = pybamm.x_average(j_inner)
        j_outer_av = pybamm.x_average(j_outer)
        j_sei = j_inner + j_outer

        variables = {
            f"Inner {self.reaction_name}interfacial current density [A.m-2]": j_inner,
            f"X-averaged inner {self.reaction_name}"
            "interfacial current density [A.m-2]": j_inner_av,
            f"Outer {self.reaction_name}interfacial current density [A.m-2]": j_outer,
            f"X-averaged outer {self.reaction_name}"
            "interfacial current density [A.m-2]": j_outer_av,
            f"{self.reaction_name}interfacial current density [A.m-2]": j_sei,
        }

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction_name}"
                    "interfacial current density [A.m-2]": j_sei_av,
                }
            )

        return variables
