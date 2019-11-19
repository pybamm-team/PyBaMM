#
# Lead-acid interface classes
#
import pybamm
from .base_interface import BaseInterface
from . import inverse_kinetics, kinetics


class BaseInterfaceLeadAcid(BaseInterface):
    """
    Base lead-acid interface class

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
        self.reaction_name = ""  # empty reaction name, assumed to be the main reaction

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
        # If c_e was broadcast, take only the orphan
        if isinstance(c_e, pybamm.Broadcast):
            c_e = c_e.orphans[0]

        if self.domain == "Negative":
            j0 = self.param.j0_n_S_ref * c_e

        elif self.domain == "Positive":
            c_w = self.param.c_w(c_e)
            j0 = self.param.j0_p_S_ref * c_e ** 2 * c_w

        return j0

    def _get_open_circuit_potential(self, variables):
        """
        A private function to obtain the open circuit potential and entropic change

        Parameters
        ----------
        variables: dict
            The variables in the full model.

        Returns
        -------
        ocp : :class:`pybamm.Symbol`
            The open-circuit potential
        dUdT : :class:`pybamm.Symbol`
            The entropic change in open-circuit potential due to temperature

        """

        c_e = variables[self.domain + " electrolyte concentration"]
        # If c_e was broadcast, take only the orphan
        if isinstance(c_e, pybamm.Broadcast):
            c_e = c_e.orphans[0]

        if self.domain == "Negative":
            ocp = self.param.U_n(c_e, self.param.T_init)
        elif self.domain == "Positive":
            ocp = self.param.U_p(c_e, self.param.T_init)

        dUdT = pybamm.Scalar(0)

        return ocp, dUdT

    def _get_number_of_electrons_in_reaction(self):
        if self.domain == "Negative":
            ne = self.param.ne_n
        elif self.domain == "Positive":
            ne = self.param.ne_p
        return ne


class ButlerVolmer(BaseInterfaceLeadAcid, kinetics.ButlerVolmer):
    """
    Extends :class:`BaseInterfaceLeadAcid` (for exchange-current density, etc) and
    :class:`kinetics.ButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class FirstOrderButlerVolmer(BaseInterfaceLeadAcid, kinetics.FirstOrderButlerVolmer):
    """
    Extends :class:`BaseInterfaceLeadAcid` (for exchange-current density, etc) and
    :class:`kinetics.FirstOrderButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseFirstOrderKinetics(
    BaseInterfaceLeadAcid, inverse_kinetics.BaseInverseFirstOrderKinetics
):
    """
    Extends :class:`BaseInterfaceLeadAcid` (for exchange-current density, etc) and
    :class:`kinetics.BaseInverseFirstOrderKinetics` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseButlerVolmer(BaseInterfaceLeadAcid, inverse_kinetics.InverseButlerVolmer):
    """
    Extends :class:`BaseInterfaceLeadAcid` (for exchange-current density, etc) and
    :class:`inverse_kinetics.InverseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
