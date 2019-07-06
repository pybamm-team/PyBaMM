#
# Lead-acid interface classes
#
import pybamm
from .base_interface import BaseInterface
from . import kinetics


class BaseInterfaceLeadAcid(BaseInterface, pybamm.lead_acid.BaseModel):
    """
    Base lead-acid interface class

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    and :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_exchange_current_density(self, variables):
        """
        A private function to obtain the exchange current density for a lead acid
        deposition reaction.

        Parameters
        ----------
        variables: dict
        `   The variables in the full model.

        Returns
        -------
        j0 : :class: `pybamm.Symbol`
            The exchange current density.
        """
        c_e = variables[self.domain + " electrolyte concentration"]

        if self.domain == "Negative":
            j0 = self.param.j0_n_S_ref * c_e

        elif self.domain == "Positive":
            c_w = self.param.c_w(c_e)
            j0 = self.param.j0_p_S_ref * c_e ** 2 * c_w

        return j0

    def _get_standard_ocp_variables(self, variables):
        """
        A private function to obtain the open circuit potential and
        related standard variables.

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            The concentration in the electrolyte.

        Returns
        -------
        variables : dict
            The variables dictionary including the open circuit potentials
            and related standard variables.
        """

        c_e = variables[self.domain + " electrolyte concentration"]
        if self.domain == "Negative":
            ocp = self.param.U_n(c_e)
            ocp_dim = self.param.U_n_ref + self.param.potential_scale * ocp
        elif self.domain == "Positive":
            ocp = self.param.U_p(c_e)
            ocp_dim = self.param.U_p_ref + self.param.potential_scale * ocp

        ocp_av = pybamm.average(ocp)
        ocp_av_dim = pybamm.average(ocp_dim)

        variables = {
            self.domain + " electrode open circuit potential": ocp,
            self.domain + " electrode open circuit potential [V]": ocp_dim,
            "Average "
            + self.domain.lower()
            + " electrode open circuit potential": ocp_av,
            "Average "
            + self.domain.lower()
            + " electrode open circuit potential [V]": ocp_av_dim,
        }

        return variables


class ButlerVolmer(BaseInterfaceLeadAcid, kinetics.BaseButlerVolmer):
    """
    Extends :class:`BaseInterfaceLeadIon` (for exchange-current density, etc) and
    :class:`kinetics.BaseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseButlerVolmer(BaseInterfaceLeadAcid, kinetics.BaseInverseButlerVolmer):
    """
    Extends :class:`BaseInterfaceLeadAcid` (for exchange-current density, etc) and
    :class:`kinetics.BaseInverseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
