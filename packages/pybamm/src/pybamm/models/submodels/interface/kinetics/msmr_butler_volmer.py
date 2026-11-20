#
# Bulter volmer class for the MSMR formulation
#

import pybamm

from .base_kinetics import BaseKinetics


class MSMRButlerVolmer(BaseKinetics):
    """
    Submodel which implements the forward Butler-Volmer equation in the MSMR
    formulation in which the interfacial current density is summed over all
    reactions.

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def _get_exchange_current_density_by_reaction(self, index):
        """
        A private function to obtain the exchange current density for each reaction
        in the MSMR formulation.

        Parameters
        ----------
        index : int
            The index of the reaction.

        Returns
        -------
        j0 : :class: `pybamm.Symbol`
            The exchange current density.
        """
        domain = self.domain
        x_average = self._x_average_inputs
        c_e = self._coupled_electrode_variable(
            "electrolyte concentration [mol.m-3]", x_average
        )
        T = self._coupled_electrode_variable("electrode temperature [K]", x_average)

        # For "particle-size distribution" submodels, take distribution version
        # of the open-circuit potential that depends on particle size.
        if getattr(self.options, domain)["particle size"] == "distribution":
            ocp = self._coupled_electrode_variable(
                "electrode open-circuit potential distribution [V]",
                x_average,
                size=True,
            )
            # broadcast c_e, T onto "particle size"
            c_e = pybamm.PrimaryBroadcast(c_e, [f"{domain} particle size"])
            T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
        else:
            ocp = self._coupled_electrode_variable(
                "electrode open-circuit potential [V]", x_average
            )

        return self.phase_param.j0_j(c_e, ocp, T, index)

    def _get_standard_exchange_current_by_reaction_variables(self, j0, index):
        domain = self.domain
        # Size average. For j0 variables that depend on particle size, see
        # "_get_standard_size_distribution_exchange_current_variables"
        if j0.domain in [["negative particle size"], ["positive particle size"]]:
            j0 = pybamm.size_average(j0)
        # Average, and broadcast if necessary
        j0_av = pybamm.x_average(j0)

        # X-average, and broadcast if necessary
        if j0.domain == []:
            j0 = pybamm.FullBroadcast(j0, f"{domain} electrode", "current collector")
        elif j0.domain == ["current collector"]:
            j0 = pybamm.PrimaryBroadcast(j0, f"{domain} electrode")

        d = domain[0]
        variables = {
            f"j0_{d}_{index} [A.m-2]": j0,
            f"X-averaged j0_{d}_{index} [A.m-2]": j0_av,
        }

        return variables

    def _get_kinetics_by_reaction(self, j0, ne, eta_r, T, u, index):
        alpha = self.phase_param.alpha_bv_j(T, index)
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        return (
            u
            * j0
            * (
                pybamm.exp(ne * (1 - alpha) * Feta_RT)
                - pybamm.exp(-ne * alpha * Feta_RT)
            )
        )

    def _get_standard_icd_by_reaction_variables(self, j, index):
        domain = self.domain
        j.print_name = f"j_{domain[0]}"

        # Size average. For j variables that depend on particle size, see
        # "_get_standard_size_distribution_interfacial_current_variables"
        if j.domain in [["negative particle size"], ["positive particle size"]]:
            j = pybamm.size_average(j)
        # Average, and broadcast if necessary
        j_av = pybamm.x_average(j)
        if j.domain == []:
            j = pybamm.FullBroadcast(j, f"{domain} electrode", "current collector")
        elif j.domain == ["current collector"]:
            j = pybamm.PrimaryBroadcast(j, f"{domain} electrode")

        d = domain[0]
        variables = {
            f"j_{d}_{index} [A.m-2]": j,
            f"X-averaged j_{d}_{index} [A.m-2]": j_av,
        }

        return variables
