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
        param = self.param

        if isinstance(self, pybamm.sei.NoSEI):
            # Set concentration scales to one for the "no SEI" model so that
            # they are not required by parameter values in general
            c_scale = 1
            c_outer_scale = 1
            v_bar = 1
        elif self.reaction_loc == "interface":
            # scales in mol/m2 (c is an interfacial quantity)
            c_scale = param.L_sei_0_dim / param.V_bar_inner_dimensional
            c_outer_scale = param.L_sei_0_dim / param.V_bar_outer_dimensional
            v_bar = param.v_bar
        else:
            # scales in mol/m3 (c is a bulk quantity)
            c_scale = (
                param.L_sei_0_dim
                * param.n.prim.a_typ
                / param.V_bar_inner_dimensional
            )
            c_outer_scale = (
                param.L_sei_0_dim
                * param.n.prim.a_typ
                / param.V_bar_outer_dimensional
            )
            v_bar = param.v_bar

        variables = {
            f"Inner {self.reaction} concentration": c_inner,
            f"Outer {self.reaction} concentration": c_outer,
        }

        if self.reaction_loc == "interface":
            variables.update({
                f"Inner {self.reaction} concentration [mol.m-2]": c_inner * c_scale,
                f"Outer {self.reaction} concentration [mol.m-2]": c_outer
                * c_outer_scale,
            })
        else:
            c_inner_av = pybamm.x_average(c_inner)
            c_outer_av = pybamm.x_average(c_outer)
            variables.update({
                f"X-averaged inner {self.reaction} concentration": c_inner_av,
                f"X-averaged inner {self.reaction} concentration [mol.m-3]": c_inner_av
                * c_scale,
                f"X-averaged outer {self.reaction} concentration": c_outer_av,
                f"X-averaged outer {self.reaction} concentration [mol.m-3]": c_outer_av
                * c_scale,
            })
        # Get variables related to the total concentration
        c_sei = c_inner + c_outer / v_bar
        variables.update(self._get_standard_total_concentration_variables(c_sei))

        return variables

    def _get_standard_total_concentration_variables(self, c_sei):
        """Update variables related to total SEI concentration."""
        param = self.param

        if isinstance(self, pybamm.sei.NoSEI):
            # Set concentration scale to one for the "no SEI" model so that
            # they are not required by parameter values in general
            c_scale = 1
            z_sei = 1
            c_inner_0 = 0
            c_outer_0 = 0
        elif self.reaction_loc == "interface":
            # scale in mol/m2 (c is an interfacial quantity)
            c_scale = param.L_sei_0_dim / param.V_bar_inner_dimensional
            z_sei = param.z_sei
            # Set scales for the "EC Reaction Limited" model
            if self.options["SEI"] == "ec reaction limited":
                c_inner_0 = 0
                c_outer_0 = 1
            else:
                c_inner_0 = param.L_inner_0
                c_outer_0 = param.L_outer_0
        else:
            # scale in mol/m3 (c is a bulk quantity)
            c_scale = (
                param.L_sei_0_dim
                * param.n.prim.a_typ
                / param.V_bar_inner_dimensional
            )
            z_sei = param.z_sei
            # Set scales for the "EC Reaction Limited" model
            if self.options["SEI"] == "ec reaction limited":
                c_inner_0 = 0
                c_outer_0 = 1
            else:
                c_inner_0 = param.L_inner_0
                c_outer_0 = param.L_outer_0

        c_sei_xav = pybamm.x_average(c_sei)
        c_sei_av = pybamm.yz_average(c_sei_xav)

        # Calculate change in SEI concentration with respect to initial state
        if self.reaction == "SEI":
            delta_c_sei = c_sei_av - (c_inner_0 + c_outer_0)
        elif self.reaction == "SEI on cracks":
            delta_c_sei = c_sei_av - (c_inner_0 + c_outer_0) / 10000

        # Q_sei in mol
        if self.reaction_loc == "interface":
            L_n = 1
        else:
            L_n = param.n.L

        Q_sei = z_sei * delta_c_sei * c_scale * L_n * param.L_y * param.L_z

        variables = {
            f"{self.reaction} concentration": c_sei,
            f"{self.reaction} concentration [mol.m-3]": c_sei * c_scale,
            f"Total {self.reaction} concentration": c_sei,
            f"Total {self.reaction} concentration [mol.m-3]": c_sei * c_scale,
            f"Loss of lithium to {self.reaction} [mol]": Q_sei,
            f"Loss of capacity to {self.reaction} [A.h]": Q_sei * param.F / 3600,
        }

        if self.reaction_loc == "interface":
            variables.update({
                f"{self.reaction} concentration [mol.m-2]": c_sei * c_scale,
                f"Total {self.reaction} concentration [mol.m-2]": c_sei * c_scale,
            })
        else:
            variables.update(
                {
                    f"X-averaged {self.reaction} concentration": c_sei_xav,
                    f"X-averaged {self.reaction} concentration [mol.m-3]": c_sei_xav
                    * c_scale,
                    f"X-averaged total {self.reaction} concentration": c_sei_xav,
                    f"X-averaged total {self.reaction} concentration [mol.m-3]":
                    c_sei_xav * c_scale,
                }
            )

        return variables

    def _get_standard_thickness_variables(self, variables):
        """Update variables related to the SEI thickness."""
        param = self.param

        if isinstance(self, pybamm.sei.NoSEI):
            # Set scales to one for the "no SEI" model so that they are
            # not required by parameter values in general
            L_scale = 1
            R_sei_dim = 1
        else:
            L_scale = param.L_sei_0_dim
            R_sei_dim = param.R_sei_dimensional

        if self.reaction == "SEI":
            c_inner = variables["Inner SEI concentration"]
            c_outer = variables["Outer SEI concentration"]

            L_inner = c_inner  # inner SEI thickness
            L_outer = c_outer  # outer SEI thickness
            L_sei = L_inner + L_outer  # SEI thickness

            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)
            L_sei_av = pybamm.x_average(L_sei)

            variables.update(
                {
                    "Inner SEI thickness": L_inner,
                    "Inner SEI thickness [m]": L_inner * L_scale,
                    "Outer SEI thickness": L_outer,
                    "Outer SEI thickness [m]": L_outer * L_scale,
                    "SEI thickness": L_sei,
                    "SEI thickness [m]": L_sei * L_scale,
                    "Total SEI thickness": L_sei,
                    "Total SEI thickness [m]": L_sei * L_scale,
                }
            )
            if self.reaction_loc != "interface":
                L_inner_av = pybamm.x_average(L_inner)
                L_outer_av = pybamm.x_average(L_outer)
                L_sei_av = pybamm.x_average(L_sei)
                variables.update(
                    {
                        "X-averaged inner SEI thickness": L_inner_av,
                        "X-averaged inner SEI thickness [m]": L_inner_av * L_scale,
                        "X-averaged outer SEI thickness": L_outer_av,
                        "X-averaged outer SEI thickness [m]": L_outer_av * L_scale,
                        "X-averaged SEI thickness": L_sei_av,
                        "X-averaged SEI thickness [m]": L_sei_av * L_scale,
                        "X-averaged total SEI thickness": L_sei_av,
                        "X-averaged total SEI thickness [m]": L_sei_av * L_scale,
                    }
                )
                if self.reaction == "SEI":
                    variables.update({
                        f"X-averaged {self.domain.lower()} electrode resistance "
                        "[Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
                    })

        # Thickness variables are handled slightly differently for SEI on cracks
        elif self.reaction == "SEI on cracks":
            c_inner_cr = variables["Inner SEI on cracks concentration"]
            c_outer_cr = variables["Outer SEI on cracks concentration"]
            roughness = variables[self.domain + " electrode roughness ratio"]

            L_inner_cr = c_inner_cr / (roughness - 1)  # inner SEI on cracks thickness
            L_outer_cr = c_outer_cr / (roughness - 1)  # outer SEI on cracks thickness
            L_SEI_cr = L_inner_cr + L_outer_cr  # SEI on cracks thickness

            L_inner_cr_av = pybamm.x_average(L_inner_cr)
            L_outer_cr_av = pybamm.x_average(L_outer_cr)
            L_SEI_cr_av = pybamm.x_average(L_SEI_cr)

            variables.update(
                {
                    "Inner SEI on cracks thickness": L_inner_cr,
                    "Inner SEI on cracks thickness [m]": L_inner_cr * L_scale,
                    "X-averaged inner SEI on cracks thickness": L_inner_cr_av,
                    "X-averaged inner SEI on cracks thickness [m]": L_inner_cr_av
                    * L_scale,
                    "Outer SEI on cracks thickness": L_outer_cr,
                    "Outer SEI on cracks thickness [m]": L_outer_cr * L_scale,
                    "X-averaged outer SEI on cracks thickness": L_outer_cr_av,
                    "X-averaged outer SEI on cracks thickness [m]": L_outer_cr_av
                    * L_scale,
                    "SEI on cracks thickness": L_SEI_cr,
                    "SEI on cracks thickness [m]": L_SEI_cr * L_scale,
                    "X-averaged SEI on cracks thickness": L_SEI_cr_av,
                    "X-averaged SEI on cracks thickness [m]": L_SEI_cr_av * L_scale,
                    "Total SEI on cracks thickness": L_SEI_cr,
                    "Total SEI on cracks thickness [m]": L_SEI_cr * L_scale,
                    "X-averaged total SEI on cracks thickness": L_SEI_cr_av,
                    "X-averaged total SEI on cracks thickness [m]": L_SEI_cr_av
                    * L_scale,
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
        j_scale = self.param.n.prim.j_scale
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        variables = {
            f"Inner {self.reaction} interfacial current density": j_inner,
            f"Inner {self.reaction} interfacial current density [A.m-2]": j_inner
            * j_scale,
            f"X-averaged inner {self.reaction} interfacial current density": j_i_av,
            f"X-averaged inner {self.reaction} "
            "interfacial current density [A.m-2]": j_i_av * j_scale,
            f"Outer {self.reaction} interfacial current density": j_outer,
            f"Outer {self.reaction} interfacial current density [A.m-2]": j_outer
            * j_scale,
            f"X-averaged outer {self.reaction} interfacial current density": j_o_av,
            f"X-averaged outer {self.reaction} "
            "interfacial current density [A.m-2]": j_o_av * j_scale,
        }

        j_sei = j_inner + j_outer
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def _get_standard_total_reaction_variables(self, j_sei):
        """Update variables related to total SEI interfacial current density."""
        j_scale = self.param.n.prim.j_scale

        variables = {
            f"{self.reaction} interfacial current density": j_sei,
            f"{self.reaction} interfacial current density [A.m-2]": j_sei * j_scale,
        }

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {self.reaction} interfacial current density": j_sei_av,
                    f"X-averaged {self.reaction} "
                    "interfacial current density [A.m-2]": j_sei_av * j_scale,
                }
            )

        return variables
