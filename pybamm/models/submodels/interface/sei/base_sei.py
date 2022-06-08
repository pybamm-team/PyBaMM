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
    phase : str
        Phase of the particle

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, options=None,phase="primary"):
        reaction = "SEI"
        domain = "Negative"
        super().__init__(param, domain, reaction, options=options,phase=phase)

    def get_coupled_variables(self, variables):
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name
        pre = self.phase_prefactor
        
        # Update some common variables
        zero_av = pybamm.PrimaryBroadcast(0, "current collector")
        zero = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        if self.reaction_loc != "interface":
            variables.update(
                {
                    f"X-averaged {domain} {phase_name}electrode SEI interfacial current "
                    "density": variables[f"X-averaged {phase_name}SEI interfacial current density"],
                    f"{Domain} {phase_name}electrode SEI interfacial current "
                    "density": variables[f"{pre}SEI interfacial current density"],
                }
            ) # Jason-whether should the value name be modified as well?
        variables.update(
            {
                "X-averaged positive electrode SEI interfacial current "
                "density": zero_av,
                "Positive electrode SEI interfacial current density": zero,
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
        # Domain = self.domain
        # domain = Domain.lower()
        phase_name = self.phase_name

        # Set length scale to one for the "no SEI" model so that it is not
        # required by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
        else:
            L_scale = param.L_sei_0_dim

        variables = {
            f"Inner {phase_name}SEI thickness": L_inner,
            f"Inner {phase_name}SEI thickness [m]": L_inner * L_scale,
            f"Outer {phase_name}SEI thickness": L_outer,
            f"Outer {phase_name}SEI thickness [m]": L_outer * L_scale,
        }

        if self.reaction_loc != "interface":
            L_inner_av = pybamm.x_average(L_inner)
            L_outer_av = pybamm.x_average(L_outer)
            variables.update(
                {
                    f"X-averaged inner {phase_name}SEI thickness": L_inner_av,
                    f"X-averaged inner {phase_name}SEI thickness [m]": L_inner_av * L_scale,
                    f"X-averaged outer {phase_name}SEI thickness": L_outer_av,
                    f"X-averaged outer {phase_name}SEI thickness [m]": L_outer_av * L_scale,
                }
            )
        # Get variables related to the total thickness
        L_sei = L_inner + L_outer
        variables.update(self._get_standard_total_thickness_variables(L_sei))

        return variables

    def _get_standard_total_thickness_variables(self, L_sei):
        """Update variables related to total SEI thickness."""
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name
        pre = self.phase_prefactor
        
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
            R_sei_dim = 1
        else:
            L_scale = self.param.L_sei_0_dim
            R_sei_dim = self.param.R_sei_dimensional

        variables = {
            f"{pre}SEI thickness": L_sei,
            f"{pre}SEI thickness [m]": L_sei * L_scale,
            f"Total {phase_name}SEI thickness": L_sei,
            f"Total {phase_name}SEI thickness [m]": L_sei * L_scale,
        }
        if self.reaction_loc != "interface":
            L_sei_av = pybamm.x_average(L_sei)
            variables.update(
                {
                    f"X-averaged {phase_name}SEI thickness": L_sei_av,
                    f"X-averaged {phase_name}SEI thickness [m]": L_sei_av * L_scale,
                    f"X-averaged total {phase_name}SEI thickness": L_sei_av,
                    f"X-averaged total {phase_name}SEI thickness [m]": L_sei_av * L_scale,
                    f"X-averaged {domain} {phase_name}electrode resistance [Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
                }
            )
        return variables

    def _get_standard_concentration_variables(self, variables):
        """Update variables related to the SEI concentration."""
        param = self.param
        # Domain = self.domain
        # domain = Domain.lower()
        phase_name = self.phase_name

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
                    param.L_sei_0_dim
                    * param.n.prim.a_typ
                    / param.V_bar_inner_dimensional
                )# Jason - does n.prim needs modification?
                n_outer_scale = (
                    param.L_sei_0_dim
                    * param.n.prim.a_typ
                    / param.V_bar_outer_dimensional
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

        L_inner = variables[f"Inner {phase_name}SEI thickness"]
        L_outer = variables[f"Outer {phase_name}SEI thickness"]

        n_inner = L_inner  # inner SEI concentration
        n_outer = L_outer  # outer SEI concentration

        n_inner_av = pybamm.x_average(L_inner)
        n_outer_av = pybamm.x_average(L_outer)

        n_SEI = n_inner + n_outer / v_bar  # SEI concentration
        n_SEI_av = pybamm.yz_average(pybamm.x_average(n_SEI))
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
                f"Inner {phase_name}SEI concentration [mol.m-3]": n_inner * n_scale,
                f"X-averaged inner {phase_name}SEI concentration [mol.m-3]": n_inner_av * n_scale,
                f"Outer {phase_name}SEI concentration [mol.m-3]": n_outer * n_outer_scale,
                f"X-averaged outer {phase_name}SEI concentration [mol.m-3]": n_outer_av
                * n_outer_scale,
                f"{pre}SEI concentration [mol.m-3]": n_SEI * n_scale,
                f"X-averaged {phase_name}SEI concentration [mol.m-3]": n_SEI_av * n_scale,
                f"Loss of lithium to {phase_name}SEI [mol]": Q_sei,
                f"Loss of capacity to {phase_name}SEI [A.h]": Q_sei * self.param.F / 3600,
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
            The variables which can be derived from the SEI thicknesses.
        """
        # Domain = self.domain
        # domain = Domain.lower()
        phase_name = self.phase_name
        
        j_scale = self.param.n.prim.j_scale
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        variables = {
            f"Inner {phase_name}SEI interfacial current density": j_inner,
            f"Inner {phase_name}SEI interfacial current density [A.m-2]": j_inner * j_scale,
            f"X-averaged inner {phase_name}SEI interfacial current density": j_i_av,
            f"X-averaged inner {phase_name}SEI interfacial current density [A.m-2]": j_i_av
            * j_scale,
            f"Outer {phase_name}SEI interfacial current density": j_outer,
            f"Outer {phase_name}SEI interfacial current density [A.m-2]": j_outer * j_scale,
            f"X-averaged outer {phase_name}SEI interfacial current density": j_o_av,
            f"X-averaged outer {phase_name}SEI interfacial current density [A.m-2]": j_o_av
            * j_scale,
        }

        j_sei = j_inner + j_outer
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def _get_standard_total_reaction_variables(self, j_sei):
        """Update variables related to total SEI interfacial current density."""
        # Domain = self.domain
        # domain = Domain.lower()
        phase_name = self.phase_name
        pre = self.phase_prefactor
        
        j_scale = self.param.n.prim.j_scale

        variables = {
            f"{pre}SEI interfacial current density": j_sei,
            f"{pre}SEI interfacial current density [A.m-2]": j_sei * j_scale,
        }# Jason-should the phase_name be changed with a capital letter

        if self.reaction_loc != "interface":
            j_sei_av = pybamm.x_average(j_sei)
            variables.update(
                {
                    f"X-averaged {phase_name}SEI interfacial current density": j_sei_av,
                    f"X-averaged {phase_name}SEI interfacial current density [A.m-2]": j_sei_av
                    * j_scale,
                }
            )

        return variables
