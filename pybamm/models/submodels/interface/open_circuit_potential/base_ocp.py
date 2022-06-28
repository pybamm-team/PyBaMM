#
# Base class for open circuit potential
#
import pybamm
from ..base_interface import BaseInterface


class BaseOpenCircuitPotential(BaseInterface):
    """
    Base class for open circuit potentials

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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options)

    def _get_standard_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open circuit potential and
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
            The variables dictionary including the open circuit potentials
            and related standard variables.
        """
        Domain = self.domain
        domain = Domain.lower()
        rxn = self.reaction_name

        # Update size variables then size average.
        if ocp.domain in [["negative particle size"], ["positive particle size"]]:
            variables = self._get_standard_size_distribution_ocp_variables(ocp, dUdT)
            ocp = pybamm.size_average(ocp)
            dUdT = pybamm.size_average(dUdT)
        else:
            variables = {}

        # Average, and broadcast if necessary
        dUdT_av = pybamm.x_average(dUdT)
        ocp_av = pybamm.x_average(ocp)
        if self.half_cell and self.domain == "Negative":
            # Half-cell domain, ocp should not be broadcast
            pass
        elif ocp.domain == []:
            ocp = pybamm.FullBroadcast(
                ocp, self.domain_for_broadcast, "current collector"
            )
        elif ocp.domain == ["current collector"]:
            ocp = pybamm.PrimaryBroadcast(ocp, self.domain_for_broadcast)

        pot_scale = self.param.potential_scale
        ocp_dim = self.domain_param.U_ref + pot_scale * ocp
        ocp_av_dim = self.domain_param.U_ref + pot_scale * ocp_av

        variables.update(
            {
                f"{Domain} electrode{rxn} open circuit potential": ocp,
                f"{Domain} electrode{rxn} open circuit potential [V]": ocp_dim,
                f"X-averaged {domain} electrode{rxn} open circuit potential": ocp_av,
                f"X-averaged {domain} electrode{rxn} "
                "open circuit potential [V]": ocp_av_dim,
            }
        )
        if self.reaction in ["lithium-ion main", "lead-acid main"]:
            variables.update(
                {
                    f"{Domain} electrode entropic change": dUdT,
                    f"{Domain} electrode entropic change [V.K-1]": pot_scale
                    * dUdT
                    / self.param.Delta_T,
                    f"X-averaged {domain} electrode entropic change": dUdT_av,
                    f"X-averaged {domain} electrode entropic change [V.K-1]": pot_scale
                    * dUdT_av
                    / self.param.Delta_T,
                }
            )

        return variables

    def _get_standard_size_distribution_ocp_variables(self, ocp, dUdT):
        """
        A private function to obtain the open circuit potential and
        related standard variables when there is a distribution of particle sizes.
        """
        Domain = self.domain
        domain = Domain.lower()
        rxn = self.reaction_name

        # X-average or broadcast to electrode if necessary
        if ocp.domains["secondary"] != [f"{domain} electrode"]:
            ocp_av = ocp
            ocp = pybamm.SecondaryBroadcast(ocp, self.domain_for_broadcast)
        else:
            ocp_av = pybamm.x_average(ocp)

        if dUdT.domains["secondary"] != [f"{domain} electrode"]:
            dUdT_av = dUdT
            dUdT = pybamm.SecondaryBroadcast(dUdT, self.domain_for_broadcast)
        else:
            dUdT_av = pybamm.x_average(dUdT)

        pot_scale = self.param.potential_scale
        Delta_T = self.param.Delta_T
        ocp_dim = self.domain_param.U_ref + pot_scale * ocp
        ocp_av_dim = self.domain_param.U_ref + pot_scale * ocp_av

        variables = {
            f"{Domain} electrode{rxn} open circuit potential distribution": ocp,
            f"{Domain} electrode{rxn} open circuit potential distribution [V]": ocp_dim,
            f"X-averaged {domain} electrode{rxn} "
            "open circuit potential distribution": ocp_av,
            f"X-averaged {domain} electrode{rxn} "
            "open circuit potential distribution [V]": ocp_av_dim,
        }
        if self.reaction_name == "":
            variables.update(
                {
                    f"{Domain} electrode entropic change (size-dependent)": dUdT,
                    f"{Domain} electrode entropic change"
                    " (size-dependent) [V.K-1]": pot_scale * dUdT / Delta_T,
                    f"X-averaged {domain} electrode entropic change"
                    " (size-dependent)": dUdT_av,
                    f"X-averaged {domain} electrode entropic change"
                    " (size-dependent) [V.K-1]": pot_scale * dUdT_av / Delta_T,
                }
            )

        return variables
