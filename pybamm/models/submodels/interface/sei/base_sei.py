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

    **Extends:** :class:`pybamm.interface.BaseInterface`
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
        zero_av = pybamm.PrimaryBroadcast(0, "current collector")
        zero = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        if self.reaction_loc != "interface":
            variables.update(
                {
                    f"X-averaged negative electrode {self.reaction_name}interfacial "
                    "current density": variables[
                        f"X-averaged {self.reaction_name}interfacial current density"
                    ],
                    f"Negative electrode {self.reaction_name}interfacial current "
                    "density": variables[
                        f"{self.reaction_name}interfacial current density"
                    ],
                    f"Negative electrode {self.reaction_name}interfacial current "
                    "density [A.m-2]": variables[
                        f"{self.reaction_name}interfacial current density [A.m-2]"
                    ],
                }
            )
            variables.update(
                self._get_standard_volumetric_current_density_variables(variables)
            )

        variables.update(
            {
                f"X-averaged positive electrode {self.reaction} "
                "interfacial current density": zero_av,
                f"Positive electrode {self.reaction} "
                "interfacial current density": zero,
                f"Positive electrode {self.reaction} "
                "interfacial current density [A.m-2]": zero,
                f"X-averaged positive electrode {self.reaction} "
                "volumetric interfacial current density": zero_av,
                f"Positive electrode {self.reaction} "
                "volumetric interfacial current density": zero,
            }
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
        # Set length scale to one for the "no SEI" model so that it is not
        # required by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
        else:
            L_scale = self.phase_param.L_sei_0_dim

        variables = {
            f"Inner {self.reaction_name}thickness": L_inner,
            f"Inner {self.reaction_name}thickness [m]": L_inner * L_scale,
            f"Outer {self.reaction_name}thickness": L_outer,
            f"Outer {self.reaction_name}thickness [m]": L_outer * L_scale,
        }

        if self.reaction_loc != "interface":
            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)
            variables.update(
                {
                    f"X-averaged inner {self.reaction_name}thickness": L_inner_av,
                    f"X-averaged inner {self.reaction_name}thickness [m]": L_inner_av
                    * L_scale,
                    f"X-averaged outer {self.reaction_name}thickness": L_outer_av,
                    f"X-averaged outer {self.reaction_name}thickness [m]": L_outer_av
                    * L_scale,
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
            L_scale = 1
            R_sei_dim = 1
        else:
            L_scale = self.phase_param.L_sei_0_dim
            R_sei_dim = self.phase_param.R_sei_dimensional

        variables = {
            f"{self.reaction_name}thickness": L_sei,
            f"{self.reaction_name}[m]": L_sei * L_scale,
            f"Total {self.reaction_name}thickness": L_sei,
            f"Total {self.reaction_name}thickness [m]": L_sei * L_scale,
        }
        if self.reaction_loc != "interface":
            L_sei_av = pybamm.x_average(L_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction_name}thickness": L_sei_av,
                    f"X-averaged {self.reaction_name}thickness [m]": L_sei_av * L_scale,
                    f"X-averaged total {self.reaction_name}thickness": L_sei_av,
                    f"X-averaged total {self.reaction_name}thickness [m]": L_sei_av
                    * L_scale,
                }
            )
            if self.reaction == "SEI":
                variables.update(
                    {
                        f"X-averaged {domain} electrode resistance "
                        "[Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
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
            n_scale = 1
            n_outer_scale = 1
            v_bar = 1
            L_inner_0 = 0
            L_outer_0 = 0
            L_inner_crack_0 = 0
            L_outer_crack_0 = 0
            z_sei = 1
        else:
            if self.reaction_loc == "interface":
                # scales in mol/m2 (n is an interfacial quantity)
                n_scale = phase_param.L_sei_0_dim / phase_param.V_bar_inner_dimensional
                n_outer_scale = (
                    phase_param.L_sei_0_dim / phase_param.V_bar_outer_dimensional
                )
            else:
                # scales in mol/m3 (n is a bulk quantity)
                n_scale = (
                    phase_param.L_sei_0_dim
                    * phase_param.a_typ
                    / phase_param.V_bar_inner_dimensional
                )
                n_outer_scale = (
                    phase_param.L_sei_0_dim
                    * phase_param.a_typ
                    / phase_param.V_bar_outer_dimensional
                )
            v_bar = phase_param.v_bar
            z_sei = phase_param.z_sei
            # Set scales for the "EC Reaction Limited" models (both symmetric and
            # asymmetric)
            if self.options["SEI"].startswith("ec reaction limited"):
                L_inner_0 = 0
                L_outer_0 = 1
                L_inner_crack_0 = 0
                # Dividing by 10000 makes initial condition effectively zero
                # without triggering division by zero errors
                L_outer_crack_0 = 1 / 10000
            else:
                L_inner_0 = phase_param.L_inner_0
                L_outer_0 = phase_param.L_outer_0
                L_inner_crack_0 = phase_param.L_inner_crack_0
                L_outer_crack_0 = phase_param.L_outer_crack_0

        if self.reaction == "SEI":
            L_inner = variables[f"Inner {reaction_name}thickness"]
            L_outer = variables[f"Outer {reaction_name}thickness"]

            n_inner = L_inner  # inner SEI concentration
            n_outer = L_outer  # outer SEI concentration

            n_inner_av = pybamm.x_average(n_inner)
            n_outer_av = pybamm.x_average(n_outer)

            n_SEI = n_inner + n_outer / v_bar  # SEI concentration
            n_SEI_xav = pybamm.x_average(n_SEI)
            n_SEI_av = pybamm.yz_average(n_SEI_xav)

            # Calculate change in SEI concentration with respect to initial state
            delta_n_SEI = n_SEI_av - (L_inner_0 + L_outer_0 / v_bar)

            # Q_sei in mol
            if self.reaction_loc == "interface":
                L_n = 1
            else:
                L_n = self.param.n.L

            Q_sei = (
                z_sei * delta_n_SEI * n_scale * L_n * self.param.L_y * self.param.L_z
            )

            variables.update(
                {
                    f"Inner {reaction_name}concentration [mol.m-3]": n_inner * n_scale,
                    f"X-averaged inner {reaction_name}"
                    "concentration [mol.m-3]": n_inner_av * n_scale,
                    f"Outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer * n_outer_scale,
                    f"X-averaged outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer_av * n_outer_scale,
                    f"{reaction_name}concentration [mol.m-3]": n_SEI * n_scale,
                    f"X-averaged {reaction_name}"
                    "concentration [mol.m-3]": n_SEI_xav * n_scale,
                    f"Loss of lithium to {reaction_name}[mol]": Q_sei,
                    f"Loss of capacity to {reaction_name}[A.h]": Q_sei
                    * self.param.F
                    / 3600,
                }
            )
        # Concentration variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            L_inner_cr = variables[f"Inner {reaction_name}thickness"]
            L_outer_cr = variables[f"Outer {reaction_name}thickness"]
            roughness = variables[f"{Domain} electrode roughness ratio"]

            n_inner_cr = L_inner_cr * (roughness - 1)  # inner SEI cracks concentration
            n_outer_cr = L_outer_cr * (roughness - 1)  # outer SEI cracks concentration

            n_inner_cr_av = pybamm.x_average(n_inner_cr)
            n_outer_cr_av = pybamm.x_average(n_outer_cr)

            n_SEI_cr = n_inner_cr + n_outer_cr / v_bar  # SEI on cracks concentration
            n_SEI_cr_xav = pybamm.x_average(n_SEI_cr)
            n_SEI_cr_av = pybamm.yz_average(n_SEI_cr_xav)

            # Calculate change in SEI cracks concentration
            # Initial state depends on roughness (to avoid division by zero)
            roughness_av = pybamm.yz_average(pybamm.x_average(roughness))
            # choose an initial condition that is as close to zero to get the
            # physics right, but doesn't cause a division by zero error
            n_SEI_cr_init = (L_inner_crack_0 + L_outer_crack_0 / v_bar) * (
                roughness_av - 1
            )
            delta_n_SEI_cr = n_SEI_cr_av - n_SEI_cr_init

            # Q_sei_cr in mol
            Q_sei_cr = (
                z_sei
                * delta_n_SEI_cr
                * n_scale
                * self.param.n.L
                * self.param.L_y
                * self.param.L_z
            )

            variables.update(
                {
                    f"Inner {reaction_name}"
                    "concentration [mol.m-3]": n_inner_cr * n_scale,
                    f"X-averaged inner {reaction_name}"
                    "concentration [mol.m-3]": n_inner_cr_av * n_scale,
                    f"Outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer_cr * n_outer_scale,
                    f"X-averaged outer {reaction_name}"
                    "concentration [mol.m-3]": n_outer_cr_av * n_outer_scale,
                    f"{reaction_name}" "concentration [mol.m-3]": n_SEI_cr * n_scale,
                    f"X-averaged {reaction_name}"
                    "concentration [mol.m-3]": n_SEI_cr_xav * n_scale,
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
        j_scale = self.phase_param.j_scale
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        variables = {
            f"Inner {self.reaction_name}interfacial current density": j_inner,
            f"Inner {self.reaction_name}interfacial current density [A.m-2]": j_inner
            * j_scale,
            f"X-averaged inner {self.reaction_name}interfacial current density": j_i_av,
            f"X-averaged inner {self.reaction_name}"
            "interfacial current density [A.m-2]": j_i_av * j_scale,
            f"Outer {self.reaction_name}interfacial current density": j_outer,
            f"Outer {self.reaction_name}interfacial current density [A.m-2]": j_outer
            * j_scale,
            f"X-averaged outer {self.reaction_name}interfacial current density": j_o_av,
            f"X-averaged outer {self.reaction_name}"
            "interfacial current density [A.m-2]": j_o_av * j_scale,
        }

        j_sei = j_inner + j_outer
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def _get_standard_total_reaction_variables(self, j_sei):
        """Update variables related to total SEI interfacial current density."""
        j_scale = self.phase_param.j_scale

        variables = {
            f"{self.reaction_name}interfacial current density": j_sei,
            f"{self.reaction_name}interfacial current density [A.m-2]": j_sei * j_scale,
        }

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction_name}"
                    "interfacial current density": j_sei_av,
                    f"X-averaged {self.reaction_name}"
                    "interfacial current density [A.m-2]": j_sei_av * j_scale,
                }
            )

        return variables
