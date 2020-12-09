#
# Base class for Li plating models.
#
import pybamm
from ..base_interface import BaseInterface


class BasePlating(BaseInterface):
    """Base class for Li plating models.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms
    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        if domain == "Positive" and not isinstance(self, pybamm.li_plating.NoPlating):
            raise NotImplementedError(
                "Li plating models are not implemented for the positive electrode"
            )
        reaction = "Li plating"
        super().__init__(param, domain, reaction)

    def _get_standard_concentration_variables(self, c_plated_Li, c_dead_Li):
        """
        A private function to obtain the standard variables which
        can be derived from the local plated Li concentration.
        Parameters
        ----------
        c_plated_Li : :class:`pybamm.Symbol`
            The plated Li concentration.
        Returns
        -------
        variables : dict
            The variables which can be derived from the plated Li thickness.
        """
        param = self.param

        # Set scales to one for the "no plating" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.li_plating.NoPlating):
            c_scale = 1
            L_scale = 1
        else:
            c_scale = param.c_e_typ
            L_scale = param.V_bar_plated_Li * c_scale / param.a_n_typ

        c_plated_Li_av = pybamm.x_average(c_plated_Li)
        L_plated_Li = c_plated_Li  # plated Li thickness
        L_plated_Li_av = pybamm.x_average(L_plated_Li)
        Q_plated_Li = c_plated_Li_av * param.L_n * param.L_y * param.L_z

        c_dead_Li_av = pybamm.x_average(c_dead_Li)
        Q_dead_Li = c_dead_Li_av * param.L_n * param.L_y * param.L_z

        domain = self.domain.lower() + " electrode"
        Domain = domain.capitalize()

        variables = {
            f"{Domain} Li plating concentration": c_plated_Li,
            f"{Domain} Li plating concentration [mol.m-3]": c_plated_Li * c_scale,
            f"{Domain} X-averaged Li plating concentration": c_plated_Li_av,
            f"X-averaged {domain} Li plating concentration [mol.m-3]": c_plated_Li_av
            * c_scale,
            f"{Domain} dead Li concentration": c_dead_Li,
            f"{Domain} dead Li concentration [mol.m-3]": c_dead_Li * c_scale,
            f"{Domain} X-averaged dead Li concentration": c_dead_Li_av,
            f"X-averaged {domain} dead Li concentration [mol.m-3]": c_dead_Li_av
            * c_scale,
            f"{Domain} Li plating thickness [m]": L_plated_Li * L_scale,
            f"X-averaged {domain} Li plating thickness [m]": L_plated_Li_av * L_scale,
            f"Loss of lithium to {domain} Li plating [mol]": (Q_plated_Li + Q_dead_Li)
            * c_scale,
            f"Loss of capacity to {domain} Li plating [A.h]": (Q_plated_Li + Q_dead_Li)
            * c_scale * param.F / 3600,
        }

        return variables

    def _get_standard_reaction_variables(self, j_stripping):
        """
            A private function to obtain the standard variables which
            can be derived from the Li stripping interfacial reaction current
            Parameters
            ----------
            j_stripping : :class:`pybamm.Symbol`
                The net Li stripping interfacial reaction current.
            Returns
            -------
            variables : dict
                The variables which can be derived from the plated Li thickness.
        """
        # Set scales to one for the "no plating" model so that they are not required
        # by parameter values in general
        param = self.param
        if self.domain == "Negative":
            j_scale = param.j_scale_n
        elif self.domain == "Positive":
            j_scale = param.j_scale_p
        j_stripping_av = pybamm.x_average(j_stripping)

        domain = self.domain.lower() + " electrode"
        Domain = domain.capitalize()

        variables = {
            f"{Domain} Li plating interfacial current density": j_stripping,
            f"{Domain} Li plating interfacial "
            f"current density [A.m-2]": j_stripping * j_scale,
            f"X-averaged {domain} Li plating "
            f"interfacial current density": j_stripping_av,
            f"X-averaged {domain} Li plating "
            f"interfacial current density [A.m-2]": j_stripping_av * j_scale,
        }

        return variables
