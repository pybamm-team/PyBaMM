#
# Base class for lithium plating models.
#
import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


class BasePlating(BaseInterface):
    """Base class for lithium plating models, from :footcite:t:`OKane2020` and
    :footcite:t:`OKane2022`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None, phase="primary"):
        reaction = "lithium plating"
        super().__init__(param, domain, reaction, options=options, phase=phase)

    def get_coupled_variables(self, variables):
        # Update some common variables
        domain, Domain = self.domain_Domain

        j_plating_av = variables[
            f"X-averaged {domain} electrode {self.phase_name}lithium plating "
            "interfacial current density [A.m-2]"
        ]
        j_plating = variables[
            f"{Domain} electrode {self.phase_name}lithium plating "
            "interfacial current density [A.m-2]"
        ]
        variables.update(
            {
                f"X-averaged {domain} electrode {self.phase_name}lithium plating "
                "interfacial current density [A.m-2]": j_plating_av,
                f"{Domain} electrode {self.phase_name}lithium plating "
                "interfacial current density [A.m-2]": j_plating,
            }
        )

        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        return variables

    def _get_standard_concentration_variables(self, c_plated_Li, c_dead_Li):
        """
        A private function to obtain the standard variables which
        can be derived from the local plated lithium concentration.
        Parameters
        ----------
        c_plated_Li : :class:`pybamm.Symbol`
            The plated lithium concentration.
        Returns
        -------
        variables : dict
            The variables which can be derived from the plated lithium thickness.
        """
        phase_name = self.phase_name
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain

        # Set scales to one for the "no plating" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.lithium_plating.NoPlating):
            c_to_L = 1
            L_k = 1
        elif domain == "negative":
            c_to_L = self.param.V_bar_Li / phase_param.a_typ
            L_k = self.param.n.L
        elif domain == "positive":
            c_to_L = self.param.V_bar_Li / phase_param.a_typ
            L_k = self.param.p.L

        c_plated_Li_av = pybamm.x_average(c_plated_Li)
        L_plated_Li = c_plated_Li * c_to_L  # plated Li thickness
        L_plated_Li_av = pybamm.x_average(L_plated_Li)
        Q_plated_Li = c_plated_Li_av * L_k * self.param.L_y * self.param.L_z

        c_dead_Li_av = pybamm.x_average(c_dead_Li)
        # dead Li "thickness", required by porosity submodel
        L_dead_Li = c_dead_Li * c_to_L
        L_dead_Li_av = pybamm.x_average(L_dead_Li)
        Q_dead_Li = c_dead_Li_av * L_k * self.param.L_y * self.param.L_z

        variables = {
            f"{Domain} {phase_name}lithium plating concentration "
            "[mol.m-3]": c_plated_Li,
            f"X-averaged {domain} {phase_name}lithium plating concentration "
            "[mol.m-3]": c_plated_Li_av,
            f"{Domain} {phase_name}dead lithium concentration [mol.m-3]": c_dead_Li,
            f"X-averaged {domain} {phase_name}dead lithium concentration "
            "[mol.m-3]": c_dead_Li_av,
            f"{Domain} {phase_name}lithium plating thickness [m]": L_plated_Li,
            f"X-averaged {domain} {phase_name} lithium plating thickness "
            "[m]": L_plated_Li_av,
            f"{Domain} {phase_name}dead lithium thickness [m]": L_dead_Li,
            f"X-averaged {domain} {phase_name}dead lithium thickness [m]": L_dead_Li_av,
            f"Loss of lithium to {domain} {phase_name}lithium plating " "[mol]": (
                Q_plated_Li + Q_dead_Li
            ),
            f"Loss of capacity to {domain} {phase_name}lithium plating " "[A.h]": (
                Q_plated_Li + Q_dead_Li
            )
            * self.param.F
            / 3600,
        }

        return variables

    def _get_standard_reaction_variables(self, j_stripping):
        """
        A private function to obtain the standard variables which
        can be derived from the lithum stripping interfacial reaction current
        Parameters
        ----------
        j_stripping : :class:`pybamm.Symbol`
            The net lithium stripping interfacial reaction current.
        Returns
        -------
        variables : dict
            The variables which can be derived from the plated lithium thickness.
        """
        domain, Domain = self.domain_Domain
        j_stripping_av = pybamm.x_average(j_stripping)

        variables = {
            f"{Domain} electrode {self.phase_name}lithium plating "
            "interfacial current density [A.m-2]": j_stripping,
            f"X-averaged {domain} electrode {self.phase_name}lithium plating "
            "interfacial current density [A.m-2]": j_stripping_av,
        }

        return variables
