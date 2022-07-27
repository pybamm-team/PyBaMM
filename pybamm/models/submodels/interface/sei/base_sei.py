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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, options=None, cracks=False):
        if cracks is True:
            reaction = "SEI on cracks"
        else:
            reaction = "SEI"
        domain = "Negative"
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        # Update some common variables
        zero_av = pybamm.PrimaryBroadcast(0, "current collector")
        zero = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        if self.reaction_loc != "interface":
            variables.update(
                {
                    f"X-averaged negative electrode {self.reaction} interfacial "
                    "current density": variables[
                        f"X-averaged {self.reaction} interfacial current density"
                    ],
                    f"Negative electrode {self.reaction} interfacial current "
                    "density": variables[
                        f"{self.reaction} interfacial current density"
                    ],
                    f"Negative electrode {self.reaction} interfacial current "
                    "density [A.m-2]": variables[
                        f"{self.reaction} interfacial current density [A.m-2]"
                    ],
                }
            )
            variables.update(
                self._get_standard_volumetric_current_density_variables(variables)
            )

        variables.update(
            {
                f"X-averaged positive electrode {self.reaction} interfacial current "
                "density": zero_av,
                f"Positive electrode {self.reaction} interfacial current density": zero,
                f"Positive electrode {self.reaction} interfacial current density "
                "[A.m-2]": zero,
                f"X-averaged positive electrode {self.reaction} volumetric interfacial "
                "current density": zero_av,
                f"Positive electrode {self.reaction} volumetric interfacial current "
                "density": zero,
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
        param = self.param

        # Set length scale to one for the "no SEI" model so that it is not
        # required by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
        else:
            L_scale = param.L_sei_0_dim

        variables = {
            f"Inner {self.reaction} thickness": L_inner,
            f"Inner {self.reaction} thickness [m]": L_inner * L_scale,
            f"Outer {self.reaction} thickness": L_outer,
            f"Outer {self.reaction} thickness [m]": L_outer * L_scale,
        }

        if self.reaction_loc != "interface":
            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)
            variables.update(
                {
                    f"X-averaged inner {self.reaction} thickness": L_inner_av,
                    f"X-averaged inner {self.reaction} thickness [m]": L_inner_av
                    * L_scale,
                    f"X-averaged outer {self.reaction} thickness": L_outer_av,
                    f"X-averaged outer {self.reaction} thickness [m]": L_outer_av
                    * L_scale,
                }
            )
        # Get variables related to the total thickness
        L_sei = L_inner + L_outer
        variables.update(self._get_standard_total_thickness_variables(L_sei))

        return variables

    def _get_standard_total_thickness_variables(self, L_sei):
        """Update variables related to total SEI thickness."""
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
            R_sei_dim = 1
        else:
            L_scale = self.param.L_sei_0_dim
            R_sei_dim = self.param.R_sei_dimensional

        variables = {
            f"{self.reaction} thickness": L_sei,
            f"{self.reaction} [m]": L_sei * L_scale,
            f"Total {self.reaction} thickness": L_sei,
            f"Total {self.reaction} thickness [m]": L_sei * L_scale,
        }
        if self.reaction_loc != "interface":
            L_sei_av = pybamm.x_average(L_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction} thickness": L_sei_av,
                    f"X-averaged {self.reaction} thickness [m]": L_sei_av * L_scale,
                    f"X-averaged total {self.reaction} thickness": L_sei_av,
                    f"X-averaged total {self.reaction} thickness [m]": L_sei_av
                    * L_scale,
                }
            )
            if self.reaction == "SEI":
                variables.update(
                    {
                        f"X-averaged {self.domain.lower()} electrode resistance "
                        "[Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
                    }
                )
        return variables

    def _get_standard_concentration_variables(self, variables):
        """Update variables related to the SEI concentration."""
        param = self.param

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            n_scale = 1
            n_outer_scale = 1
            v_bar = 1
            L_inner_0 = 0
            L_outer_0 = 0
            li_mols_per_sei_mols = 1
        else:
            if self.reaction_loc == "interface":
                # scales in mol/m2 (n is an interfacial quantity)
                n_scale = param.L_sei_0_dim / param.V_bar_inner_dimensional
                n_outer_scale = param.L_sei_0_dim / param.V_bar_outer_dimensional
            else:
                # scales in mol/m3 (n is a bulk quantity)
                n_scale = (
                    param.L_sei_0_dim * param.n.a_typ / param.V_bar_inner_dimensional
                )
                n_outer_scale = (
                    param.L_sei_0_dim * param.n.a_typ / param.V_bar_outer_dimensional
                )
            v_bar = param.v_bar
            # Set scales for the "EC Reaction Limited" model
            if self.options["SEI"] == "ec reaction limited":
                L_inner_0 = 0
                L_outer_0 = 1
                li_mols_per_sei_mols = 2
            else:
                L_inner_0 = param.L_inner_0
                L_outer_0 = param.L_outer_0
                li_mols_per_sei_mols = 1

        if self.reaction == "SEI":
            L_inner = variables["Inner SEI thickness"]
            L_outer = variables["Outer SEI thickness"]

            n_inner = L_inner  # inner SEI concentration
            n_outer = L_outer  # outer SEI concentration

            n_inner_av = pybamm.x_average(n_inner)
            n_outer_av = pybamm.x_average(n_outer)

            n_SEI = n_inner + n_outer / v_bar  # SEI concentration
            n_SEI_av = pybamm.yz_average(pybamm.x_average(n_SEI))

            # Calculate change in SEI concentration with respect to initial state
            delta_n_SEI = n_SEI_av - (L_inner_0 + L_outer_0 / v_bar)

            # Q_sei in mol
            if self.reaction_loc == "interface":
                L_n = 1
            else:
                L_n = self.param.n.L

            Q_sei = (
                li_mols_per_sei_mols
                * delta_n_SEI
                * n_scale
                * L_n
                * self.param.L_y
                * self.param.L_z
            )

            variables.update(
                {
                    "Inner SEI concentration [mol.m-3]": n_inner * n_scale,
                    "X-averaged inner SEI concentration [mol.m-3]": n_inner_av
                    * n_scale,
                    "Outer SEI concentration [mol.m-3]": n_outer * n_outer_scale,
                    "X-averaged outer SEI concentration [mol.m-3]": n_outer_av
                    * n_outer_scale,
                    "SEI concentration [mol.m-3]": n_SEI * n_scale,
                    "X-averaged SEI concentration [mol.m-3]": n_SEI_av * n_scale,
                    "Loss of lithium to SEI [mol]": Q_sei,
                    "Loss of capacity to SEI [A.h]": Q_sei * self.param.F / 3600,
                }
            )
        # Concentration variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            L_inner_cr = variables["Inner SEI on cracks thickness"]
            L_outer_cr = variables["Outer SEI on cracks thickness"]
            roughness = variables[self.domain + " electrode roughness ratio"]

            n_inner_cr = L_inner_cr * (roughness - 1)  # inner SEI cracks concentration
            n_outer_cr = L_outer_cr * (roughness - 1)  # outer SEI cracks concentration

            n_inner_cr_av = pybamm.x_average(n_inner_cr)
            n_outer_cr_av = pybamm.x_average(n_outer_cr)

            n_SEI_cr = n_inner_cr + n_outer_cr / v_bar  # SEI on cracks concentration
            n_SEI_cr_av = pybamm.yz_average(pybamm.x_average(n_SEI_cr))

            # Calculate change in SEI cracks concentration with respect to initial state
            rho_cr = param.n.rho_cr
            n_SEI_cr_init = 2 * rho_cr * (L_inner_0 + L_outer_0 / v_bar) / 10000
            delta_n_SEI_cr = n_SEI_cr_av - n_SEI_cr_init

            # Q_sei_cr in mol
            Q_sei_cr = (
                li_mols_per_sei_mols
                * delta_n_SEI_cr
                * n_scale
                * self.param.n.L
                * self.param.L_y
                * self.param.L_z
            )

            variables.update(
                {
                    "Inner SEI on cracks concentration [mol.m-3]": n_inner_cr * n_scale,
                    "X-averaged inner SEI on cracks concentration [mol.m-3]":
                    n_inner_cr_av * n_scale,
                    "Outer SEI on cracks concentration [mol.m-3]": n_outer_cr
                    * n_outer_scale,
                    "X-averaged outer SEI on cracks concentration [mol.m-3]":
                    n_outer_cr_av * n_outer_scale,
                    "SEI on cracks concentration [mol.m-3]": n_SEI_cr * n_scale,
                    "X-averaged SEI on cracks concentration [mol.m-3]": n_SEI_cr_av
                    * n_scale,
                    "Loss of lithium to SEI on cracks [mol]": Q_sei_cr,
                    "Loss of capacity to SEI on cracks [A.h]": Q_sei_cr
                    * self.param.F / 3600,
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
        j_scale = self.param.n.j_scale
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        variables = {
            f"Inner {self.reaction} interfacial current density": j_inner,
            f"Inner {self.reaction} interfacial current density [A.m-2]": j_inner
            * j_scale,
            f"X-averaged inner {self.reaction} interfacial current density": j_i_av,
            f"X-averaged inner {self.reaction} interfacial current density [A.m-2]":
            j_i_av * j_scale,
            f"Outer {self.reaction} interfacial current density": j_outer,
            f"Outer {self.reaction} interfacial current density [A.m-2]": j_outer
            * j_scale,
            f"X-averaged outer {self.reaction} interfacial current density": j_o_av,
            f"X-averaged outer {self.reaction} interfacial current density [A.m-2]":
            j_o_av * j_scale,
        }

        j_sei = j_inner + j_outer
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def _get_standard_total_reaction_variables(self, j_sei):
        """Update variables related to total SEI interfacial current density."""
        j_scale = self.param.n.j_scale

        variables = {
            f"{self.reaction} interfacial current density": j_sei,
            f"{self.reaction} interfacial current density [A.m-2]": j_sei * j_scale,
        }

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction} interfacial current density": j_sei_av,
                    f"X-averaged {self.reaction} interfacial current density [A.m-2]":
                    j_sei_av * j_scale,
                }
            )

        return variables
