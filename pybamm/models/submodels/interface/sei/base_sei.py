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
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        if domain == "Positive" and not isinstance(self, pybamm.sei.NoSEI):
            raise NotImplementedError(
                "SEI models are not implemented for the positive electrode"
            )
        reaction = "SEI"
        super().__init__(param, domain, reaction)

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
        domain = self.domain.lower() + " electrode"

        # Set length scale to one for the "no SEI" model so that it is not
        # required by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
        else:
            L_scale = param.L_sei_0_dim

        L_inner_av = pybamm.x_average(L_inner)
        L_outer_av = pybamm.x_average(L_outer)

        variables = {
            "Inner " + domain + " SEI thickness": L_inner,
            "Inner " + domain + " SEI thickness [m]": L_inner * L_scale,
            "X-averaged inner " + domain + " SEI thickness": L_inner_av,
            "X-averaged inner " + domain + " SEI thickness [m]": L_inner_av * L_scale,
            "Outer " + domain + " SEI thickness": L_outer,
            "Outer " + domain + " SEI thickness [m]": L_outer * L_scale,
            "X-averaged outer " + domain + " SEI thickness": L_outer_av,
            "X-averaged outer " + domain + " SEI thickness [m]": L_outer_av * L_scale,
        }

        # Get variables related to the total thickness
        L_sei = L_inner + L_outer
        variables.update(self._get_standard_total_thickness_variables(L_sei))

        return variables

    def _get_standard_total_thickness_variables(self, L_sei):
        """Update variables related to total SEI thickness."""
        domain = self.domain.lower() + " electrode"
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
            R_sei_dim = 1
        else:
            L_scale = self.param.L_sei_0_dim
            R_sei_dim = self.param.R_sei_dimensional
        L_sei_av = pybamm.x_average(L_sei)

        variables = {
            "Total " + domain + " SEI thickness": L_sei,
            "Total " + domain + " SEI thickness [m]": L_sei * L_scale,
            "X-averaged total " + domain + " SEI thickness": L_sei_av,
            "X-averaged total " + domain + " SEI thickness [m]": L_sei_av * L_scale,
            "X-averaged "
            + self.domain.lower()
            + " electrode resistance [Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
        }
        return variables

    def _get_standard_concentration_variables(self, variables):
        """Update variables related to the SEI concentration."""
        param = self.param
        domain = self.domain.lower() + " electrode"

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
            n_scale = param.L_sei_0_dim * param.a_n_typ / param.V_bar_inner_dimensional
            n_outer_scale = (
                param.L_sei_0_dim * param.a_n_typ / param.V_bar_outer_dimensional
            )
            v_bar = param.v_bar
            # Set scales for the "EC Reaction Limited" model
            if isinstance(self, pybamm.sei.EcReactionLimited):
                L_inner_0 = 0
                L_outer_0 = 1
                li_mols_per_sei_mols = 2
            else:
                L_inner_0 = param.L_inner_0
                L_outer_0 = param.L_outer_0
                li_mols_per_sei_mols = 1

        L_inner = variables["Inner " + domain + " SEI thickness"]
        L_outer = variables["Outer " + domain + " SEI thickness"]

        n_inner = L_inner  # inner SEI concentration
        n_outer = L_outer  # outer SEI concentration

        n_inner_av = pybamm.x_average(L_inner)
        n_outer_av = pybamm.x_average(L_outer)

        n_SEI = n_inner + n_outer / v_bar  # SEI concentration
        n_SEI_av = pybamm.x_average(n_SEI)
        delta_n_SEI = n_SEI_av - (L_inner_0 + L_outer_0 / v_bar)

        Q_sei = (
            li_mols_per_sei_mols
            * delta_n_SEI
            * self.param.L_n
            * self.param.L_y
            * self.param.L_z
        )

        variables.update(
            {
                "Inner " + domain + " SEI concentration [mol.m-3]": n_inner * n_scale,
                "X-averaged inner "
                + domain
                + " SEI concentration [mol.m-3]": n_inner_av * n_scale,
                "Outer "
                + domain
                + " SEI concentration [mol.m-3]": n_outer * n_outer_scale,
                "X-averaged outer "
                + domain
                + " SEI concentration [mol.m-3]": n_outer_av * n_outer_scale,
                self.domain + " SEI concentration [mol.m-3]": n_SEI * n_scale,
                "X-averaged "
                + domain
                + " SEI concentration [mol.m-3]": n_SEI_av * n_scale,
                "Loss of lithium to " + domain + " SEI [mol]": Q_sei * n_scale,
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
        if self.domain == "Negative":
            j_scale = self.param.j_scale_n
        elif self.domain == "Positive":
            j_scale = self.param.j_scale_p
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        domain = self.domain.lower() + " electrode"

        variables = {
            "Inner " + domain + " SEI interfacial current density": j_inner,
            "Inner "
            + domain
            + " SEI interfacial current density [A.m-2]": j_inner * j_scale,
            "X-averaged inner " + domain + " SEI interfacial current density": j_i_av,
            "X-averaged inner "
            + domain
            + " SEI interfacial current density [A.m-2]": j_i_av * j_scale,
            "Outer " + domain + " SEI interfacial current density": j_outer,
            "Outer "
            + domain
            + " SEI interfacial current density [A.m-2]": j_outer * j_scale,
            "X-averaged outer " + domain + " SEI interfacial current density": j_o_av,
            "X-averaged outer "
            + domain
            + " SEI interfacial current density [A.m-2]": j_o_av * j_scale,
        }

        j_sei = j_inner + j_outer
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def _get_standard_total_reaction_variables(self, j_sei):
        """Update variables related to total SEI interfacial current density."""
        if self.domain == "Negative":
            j_scale = self.param.j_scale_n
        elif self.domain == "Positive":
            j_scale = self.param.j_scale_p

        j_sei_av = pybamm.x_average(j_sei)

        domain = self.domain.lower() + " electrode"
        Domain = domain.capitalize()

        variables = {
            Domain + " SEI interfacial current density": j_sei,
            Domain + " SEI interfacial current density [A.m-2]": j_sei * j_scale,
            "X-averaged " + domain + " SEI interfacial current density": j_sei_av,
            "X-averaged "
            + domain
            + " SEI interfacial current density [A.m-2]": j_sei_av * j_scale,
        }

        return variables
