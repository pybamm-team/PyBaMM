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

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_exchange_current_density_by_reaction(self, variables, index):
        """ "
        A private function to obtain the exchange current density for each reaction
        in the MSMR formulation.

        Parameters
        ----------
        variables: dict
            The variables in the full model.

        Returns
        -------
        j0 : :class: `pybamm.Symbol`
            The exchange current density.
        """
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
        T = variables[f"{Domain} electrode temperature [K]"]

        if self.reaction == "lithium-ion main":
            # For "particle-size distribution" submodels, take distribution version
            # of c_s_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface "
                    "concentration distribution [mol.m-3]"
                ]
                # If all variables were broadcast (in "x"), take only the orphans,
                # then re-broadcast c_e
                if (
                    isinstance(c_s_surf, pybamm.Broadcast)
                    and isinstance(c_e, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    c_e = c_e.orphans[0]
                    T = T.orphans[0]

                    # as c_e must now be a scalar, re-broadcast to
                    # "current collector"
                    c_e = pybamm.PrimaryBroadcast(c_e, ["current collector"])
                # broadcast c_e, T onto "particle size"
                c_e = pybamm.PrimaryBroadcast(c_e, [f"{domain} particle size"])
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])

            else:
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
                ]
                # If all variables were broadcast, take only the orphans
                if (
                    isinstance(c_s_surf, pybamm.Broadcast)
                    and isinstance(c_e, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    c_e = c_e.orphans[0]
                    T = T.orphans[0]

            j0 = phase_param.j0_j(c_e, c_s_surf, T, index)

            # Size average. For j0 variables that depend on particle size, see
            # "_get_standard_size_distribution_exchange_current_variables"
            if j0.domain in [["negative particle size"], ["positive particle size"]]:
                j0 = pybamm.size_average(j0)
            # Average, and broadcast if necessary
            j0_av = pybamm.x_average(j0)

            # X-average, and broadcast if necessary
            if j0.domain == []:
                j0 = pybamm.FullBroadcast(
                    j0, f"{domain} electrode", "current collector"
                )
            elif j0.domain == ["current collector"]:
                j0 = pybamm.PrimaryBroadcast(j0, f"{domain} electrode")

            d = domain[0]
            variables = {
                f"j0_{d}_{index} [A.m-2]": j0,
                f"X-averaged j0_{d}_{index} [A.m-2]": j0_av,
            }

        return variables

    def _get_exchange_current_density(self, variables):
        options = self.options
        domain = self.domain
        d = domain[0]
        j0 = 0
        # Loop over all reactions
        N = int(getattr(options, domain)["number of MSMR reactions"])
        for i in range(N):
            j0 += variables[f"j0_{d}_{i} [A.m-2]"]
        return j0

    def _get_kinetics_by_reaction(self, j0, ne, eta_r, T, u, index):
        alpha = self.phase_param.alpha_bv_j(index)
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        arg_ox = ne * alpha * Feta_RT
        arg_red = -ne * (1 - alpha) * Feta_RT
        return u * j0 * (pybamm.exp(arg_ox) - pybamm.exp(arg_red))

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
