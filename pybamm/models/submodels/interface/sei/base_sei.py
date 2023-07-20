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
        if self.reaction_loc == "interface":  # c is an interfacial quantity [mol.m-2]
            variables = {
                f"Inner {self.reaction_name}concentration [mol.m-2]": c_inner,
                f"Outer {self.reaction_name}concentration [mol.m-2]": c_outer,
            }
        else:  # c is a bulk quantity [mol.m-3]
            c_inner_av = pybamm.x_average(c_inner)
            c_outer_av = pybamm.x_average(c_outer)
            variables = {
                f"Inner {self.reaction_name}concentration [mol.m-3]": c_inner,
                f"Outer {self.reaction_name}concentration [mol.m-3]": c_outer,
                f"X-averaged inner {self.reaction_name}"
                "concentration [mol.m-3]": c_inner_av,
                f"X-averaged outer {self.reaction_name}"
                "concentration [mol.m-3]": c_outer_av,
            }

        # Get variables related to the total thickness
        c_sei = c_inner + c_outer
        variables.update(self._get_standard_total_concentration_variables(c_sei))

        return variables

    def _get_standard_total_concentration_variables(self, c_sei):
        """Update variables related to total SEI concentration."""
        if self.reaction_loc == "interface":  # c is an interfacial quantity [mol.m-2]
            c_sei_av = pybamm.yz_average(c_sei)
            c_sei_0 = (
                self.phase_param.L_inner_0 / self.phase_param.V_bar_inner
                + self.phase_param.L_outer_0 / self.phase_param.V_bar_outer
            )
            L_n = 1
            variables = {
                f"{self.reaction_name}concentration [mol.m-2]": c_sei,
                f"Total {self.reaction_name}concentration [mol.m-2]": c_sei,
            }
        else:  # c is a bulk quantity [mol.m-3]
            c_xav = pybamm.x_average(c_sei)
            c_sei_av = pybamm.yz_average(c_xav)
            if isinstance(self, pybamm.sei.NoSEI):
                c_sei_0 = 0 * c_sei_av  # Set initial SEI to zero, copying domains
            else:
                c_sei_0 = self.phase_param.a_typ * (
                    self.phase_param.L_inner_0 / self.phase_param.V_bar_inner
                    + self.phase_param.L_outer_0 / self.phase_param.V_bar_outer
                )
            L_n = self.param.n.L
            variables = {
                f"{self.reaction_name}concentration [mol.m-3]": c_sei,
                f"Total {self.reaction_name}concentration [mol.m-3]": c_sei,
                f"X-averaged {self.reaction_name}concentration [mol.m-3]": c_xav,
                f"X-averaged total {self.reaction_name}concentration [mol.m-3]": c_xav,
            }
        # Calculate change in SEI concentration with respect to initial state
        # If there is no SEI, skip this step because parameters may be undefined
        if isinstance(self, pybamm.sei.NoSEI):
            z_sei = 1
            delta_c_SEI = c_sei_av
        else:
            z_sei = self.phase_param.z_sei
            if self.reaction == "SEI":
                delta_c_SEI = c_sei_av - c_sei_0
            elif self.reaction == "SEI on cracks":
                roughness_init = 1 + 2 * (
                    self.param.n.l_cr_0 * self.param.n.rho_cr * self.param.n.w_cr
                )
                c_sei_cracks_0 = (
                    (roughness_init - 1)
                    * self.phase_param.a_typ
                    * (
                        self.phase_param.L_inner_crack_0 / self.phase_param.V_bar_inner
                        + self.phase_param.L_outer_crack_0
                        / self.phase_param.V_bar_outer
                    )
                )
                delta_c_SEI = c_sei_av - c_sei_cracks_0
        # Multiply delta_n_SEI by V_n to get total moles of SEI formed
        # Multiply by z_sei to get total lithium moles consumed by SEI
        V_n = L_n * self.param.L_y * self.param.L_z
        Q_sei = delta_c_SEI * V_n * z_sei
        variables.update(
            {
                f"Loss of lithium to {self.reaction_name}[mol]": Q_sei,
                f"Loss of capacity to {self.reaction_name}[A.h]": Q_sei
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
