#
# Base class for lithium plating models.
#
import pybamm
from ..base_interface import BaseInterface


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

    def __init__(self, param, options=None):
        reaction = "lithium plating"
        domain = "negative"
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        # Update some common variables

        if self.options.electrode_types["negative"] == "porous":
            j_plating = variables["Lithium plating interfacial current density [A.m-2]"]
            j_plating_av = variables[
                "X-averaged lithium plating interfacial current density [A.m-2]"
            ]
            if self.options.negative["particle phases"] == "1":
                a = variables["Negative electrode surface area to volume ratio [m-1]"]
            else:
                a = variables[
                    "Negative electrode primary surface area to volume ratio [m-1]"
                ]
            a_j_plating = a * j_plating
            a_j_plating_av = pybamm.x_average(a_j_plating)

            variables.update(
                {
                    "Negative electrode lithium plating interfacial current "
                    "density [A.m-2]": j_plating,
                    "X-averaged negative electrode lithium plating "
                    "interfacial current density [A.m-2]": j_plating_av,
                    "Lithium plating volumetric "
                    "interfacial current density [A.m-3]": a_j_plating,
                    "X-averaged lithium plating volumetric "
                    "interfacial current density [A.m-3]": a_j_plating_av,
                }
            )

        zero_av = pybamm.PrimaryBroadcast(0, "current collector")
        zero = pybamm.FullBroadcast(0, "positive electrode", "current collector")
        variables.update(
            {
                "X-averaged positive electrode lithium plating "
                "interfacial current density [A.m-2]": zero_av,
                "X-averaged positive electrode lithium plating volumetric "
                "interfacial current density [A.m-3]": zero_av,
                "Positive electrode lithium plating "
                "interfacial current density [A.m-2]": zero,
                "Positive electrode lithium plating volumetric "
                "interfacial current density [A.m-3]": zero,
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
        param = self.param

        # Set scales to one for the "no plating" model so that they are not required
        # by parameter values in general
        if isinstance(self, pybamm.lithium_plating.NoPlating):
            c_to_L = 1
        else:
            c_to_L = param.V_bar_plated_Li / param.n.prim.a_typ

        c_plated_Li_av = pybamm.x_average(c_plated_Li)
        L_plated_Li = c_plated_Li * c_to_L  # plated Li thickness
        L_plated_Li_av = pybamm.x_average(L_plated_Li)
        Q_plated_Li = c_plated_Li_av * param.n.L * param.L_y * param.L_z

        c_dead_Li_av = pybamm.x_average(c_dead_Li)
        # dead Li "thickness", required by porosity submodel
        L_dead_Li = c_dead_Li * c_to_L
        L_dead_Li_av = pybamm.x_average(L_dead_Li)
        Q_dead_Li = c_dead_Li_av * param.n.L * param.L_y * param.L_z

        variables = {
            "Lithium plating concentration [mol.m-3]": c_plated_Li,
            "X-averaged lithium plating concentration [mol.m-3]": c_plated_Li_av,
            "Dead lithium concentration [mol.m-3]": c_dead_Li,
            "X-averaged dead lithium concentration [mol.m-3]": c_dead_Li_av,
            "Lithium plating thickness [m]": L_plated_Li,
            "X-averaged lithium plating thickness [m]": L_plated_Li_av,
            "Dead lithium thickness [m]": L_dead_Li,
            "X-averaged dead lithium thickness [m]": L_dead_Li_av,
            "Loss of lithium to lithium plating [mol]": (Q_plated_Li + Q_dead_Li),
            "Loss of capacity to lithium plating [A.h]": (Q_plated_Li + Q_dead_Li)
            * param.F
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
        # Set scales to one for the "no plating" model so that they are not required
        # by parameter values in general
        j_stripping_av = pybamm.x_average(j_stripping)

        variables = {
            "Lithium plating interfacial current density [A.m-2]": j_stripping,
            "X-averaged lithium plating "
            "interfacial current density [A.m-2]": j_stripping_av,
        }

        return variables
