#
# Base class for open-circuit potential
#
import pybamm
from ..base_interface import BaseInterface


class BaseOpenCircuitPotential(BaseInterface):
    """
    Base class for open-circuit potentials

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. See
        :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options=options, phase=phase)

    def _get_standard_ocp_variables(self, ocp_surf, ocp_bulk, dUdT):
        """
        A private function to obtain the open-circuit potential and
        related standard variables.

        Parameters
        ----------
        ocp : :class:`pybamm.Symbol`
            The open-circuit potential
        dUdT : :class:`pybamm.Symbol`
            The entropic change in ocp

        Returns
        -------
        variables : dict
            The variables dictionary including the open-circuit potentials
            and related standard variables.
        """
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        # Update size variables then size average.
        if ocp_surf.domain in [["negative particle size"], ["positive particle size"]]:
            variables = self._get_standard_size_distribution_ocp_variables(
                ocp_surf, dUdT
            )
            ocp_surf = pybamm.size_average(ocp_surf)
            dUdT = pybamm.size_average(dUdT)
        else:
            variables = {}

        # Average, and broadcast if necessary
        dUdT_av = pybamm.x_average(dUdT)
        ocp_surf_av = pybamm.x_average(ocp_surf)
        if self.options.electrode_types[domain] == "planar":
            # Half-cell domain, ocp_surf should not be broadcast
            pass
        elif ocp_surf.domain == []:
            ocp_surf = pybamm.FullBroadcast(
                ocp_surf, f"{domain} electrode", "current collector"
            )
        elif ocp_surf.domain == ["current collector"]:
            ocp_surf = pybamm.PrimaryBroadcast(ocp_surf, f"{domain} electrode")

        # Particle overpotential is the difference between the average(U(c_surf)) and
        # U(c_bulk), i.e. the overpotential due to concentration gradients in the
        # particle
        eta_particle = ocp_surf_av - ocp_bulk
        variables.update(
            {
                f"{Domain} electrode {reaction_name}"
                "open-circuit potential [V]": ocp_surf,
                f"X-averaged {domain} electrode {reaction_name}"
                "open-circuit potential [V]": ocp_surf_av,
                f"{Domain} electrode {reaction_name}"
                "bulk open-circuit potential [V]": ocp_bulk,
                f"{Domain} {reaction_name}particle concentration "
                "overpotential [V]": eta_particle,
            }
        )
        if self.reaction in ["lithium-ion main", "lead-acid main"]:
            variables.update(
                {
                    f"{Domain} electrode {reaction_name}entropic change [V.K-1]": dUdT,
                    f"X-averaged {domain} electrode {reaction_name}entropic change [V.K-1]": dUdT_av,
                }
            )

        return variables

    def _get_standard_size_distribution_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open-circuit potential and
        related standard variables when there is a distribution of particle sizes.
        """
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if ocp.domains["secondary"] != [f"{domain} electrode"]:
            ocp_av = ocp
            ocp = pybamm.SecondaryBroadcast(ocp, f"{domain} electrode")
        else:
            ocp_av = pybamm.x_average(ocp)

        if dUdT.domains["secondary"] != [f"{domain} electrode"]:
            dUdT_av = dUdT
            dUdT = pybamm.SecondaryBroadcast(dUdT, f"{domain} electrode")
        else:
            dUdT_av = pybamm.x_average(dUdT)

        variables = {
            f"{Domain} electrode {reaction_name}"
            "open-circuit potential distribution [V]": ocp,
            f"X-averaged {domain} electrode {reaction_name}"
            "open-circuit potential distribution [V]": ocp_av,
        }
        if self.reaction_name == "":
            variables.update(
                {
                    f"{Domain} electrode entropic change "
                    "(size-dependent) [V.K-1]": dUdT,
                    f"X-averaged {domain} electrode entropic change "
                    "(size-dependent) [V.K-1]": dUdT_av,
                }
            )

        return variables
