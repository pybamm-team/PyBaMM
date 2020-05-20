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
        reaction = "sei"
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
        sp = pybamm.sei_parameters

        # Set scales to one for the "no SEI" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.sei.NoSEI):
            L_scale = 1
            n_scale = 1
            n_outer_scale = 1
            v_bar = 1
        else:
            L_scale = sp.L_sei_0_dim
            n_scale = sp.L_sei_0_dim * sp.a_n / sp.V_bar_inner_dimensional
            n_outer_scale = sp.L_sei_0_dim * sp.a_n / sp.V_bar_outer_dimensional
            v_bar = sp.v_bar

        L_inner_av = pybamm.x_average(L_inner)
        L_outer_av = pybamm.x_average(L_outer)

        L_sei = L_inner + L_outer
        L_sei_av = pybamm.x_average(L_sei)

        n_inner = L_inner  # inner SEI concentration
        n_outer = L_outer  # outer SEI concentration
        n_inner_av = pybamm.x_average(L_inner)
        n_outer_av = pybamm.x_average(L_outer)

        n_SEI = n_inner + n_outer / v_bar  # SEI concentration
        n_SEI_av = pybamm.x_average(n_SEI)

        Q_sei = n_SEI_av * self.param.L_n * self.param.L_y * self.param.L_z

        domain = self.domain.lower() + " electrode"

        variables = {
            "Inner " + domain + " sei thickness": L_inner,
            "Inner " + domain + " sei thickness [m]": L_inner * L_scale,
            "X-averaged inner " + domain + " sei thickness": L_inner_av,
            "X-averaged inner " + domain + " sei thickness [m]": L_inner_av * L_scale,
            "Outer " + domain + " sei thickness": L_outer,
            "Outer " + domain + " sei thickness [m]": L_outer * L_scale,
            "X-averaged outer " + domain + " sei thickness": L_outer_av,
            "X-averaged outer " + domain + " sei thickness [m]": L_outer_av * L_scale,
            "Total " + domain + " sei thickness": L_sei,
            "Total " + domain + " sei thickness [m]": L_sei * L_scale,
            "X-averaged total " + domain + " sei thickness": L_sei_av,
            "X-averaged total " + domain + " sei thickness [m]": L_sei_av * L_scale,
            "Inner " + domain + " sei concentration [mol.m-3]": n_inner * n_scale,
            "X-averaged inner "
            + domain
            + " sei concentration [mol.m-3]": n_inner_av * n_scale,
            "Outer " + domain + " sei concentration [mol.m-3]": n_outer * n_outer_scale,
            "X-averaged outer "
            + domain
            + " sei concentration [mol.m-3]": n_outer_av * n_outer_scale,
            self.domain + " sei concentration [mol.m-3]": n_SEI * n_scale,
            "X-averaged " + domain + " sei concentration [mol.m-3]": n_SEI_av * n_scale,
            "Loss of lithium to " + domain + " sei [mol]": Q_sei * n_scale,
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
                The variables which can be derived from the SEI thicknesses.
        """
        if self.domain == "Negative":
            j_scale = self.param.interfacial_current_scale_n
        elif self.domain == "Positive":
            j_scale = self.param.interfacial_current_scale_p
        j_i_av = pybamm.x_average(j_inner)
        j_o_av = pybamm.x_average(j_outer)

        j_sei = j_inner + j_outer
        j_sei_av = pybamm.x_average(j_sei)

        domain = self.domain.lower() + " electrode"
        Domain = domain.capitalize()

        variables = {
            "Inner " + domain + " sei interfacial current density": j_inner,
            "Inner "
            + domain
            + " sei interfacial current density [A.m-2]": j_inner * j_scale,
            "X-averaged inner " + domain + " sei interfacial current density": j_i_av,
            "X-averaged inner "
            + domain
            + " sei interfacial current density [A.m-2]": j_i_av * j_scale,
            "Outer " + domain + " sei interfacial current density": j_outer,
            "Outer "
            + domain
            + " sei interfacial current density [A.m-2]": j_outer * j_scale,
            "X-averaged outer " + domain + " sei interfacial current density": j_o_av,
            "X-averaged outer "
            + domain
            + " sei interfacial current density [A.m-2]": j_o_av * j_scale,
            Domain + " sei interfacial current density": j_sei,
            Domain + " sei interfacial current density [A.m-2]": j_sei * j_scale,
            "X-averaged " + domain + " sei interfacial current density": j_sei_av,
            "X-averaged "
            + domain
            + " sei interfacial current density [A.m-2]": j_sei_av * j_scale,
        }

        return variables
