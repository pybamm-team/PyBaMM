#
# Lithium-ion interface classes
#
from .base_interface import BaseInterface
from . import inverse_kinetics, kinetics


class BaseInterfaceLithiumIon(BaseInterface):
    """
    Base lthium-ion interface class

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
        A private function to obtain the exchange current density for a lithium-ion
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
        c_s_surf = variables[self.domain + " particle surface concentration"]
        c_e = variables[self.domain + " electrolyte concentration"]
        T = variables[self.domain + " electrode temperature"]

        if self.domain == "Negative":
            prefactor = self.param.m_n(T) / self.param.C_r_n

        elif self.domain == "Positive":
            prefactor = self.param.gamma_p * self.param.m_p(T) / self.param.C_r_p

        j0 = prefactor * (
            c_e ** (1 / 2) * c_s_surf ** (1 / 2) * (1 - c_s_surf) ** (1 / 2)
        )

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
        c_s_surf = variables[self.domain + " particle surface concentration"]
        T = variables[self.domain + " electrode temperature"]

        if self.domain == "Negative":
            ocp = self.param.U_n(c_s_surf, T)
            dUdT = self.param.dUdT_n(c_s_surf)

        elif self.domain == "Positive":
            ocp = self.param.U_p(c_s_surf, T)
            dUdT = self.param.dUdT_p(c_s_surf)

        return ocp, dUdT

    def _get_number_of_electrons_in_reaction(self):
        if self.domain == "Negative":
            ne = self.param.ne_n
        elif self.domain == "Positive":
            ne = self.param.ne_p
        return ne


class ButlerVolmer(BaseInterfaceLithiumIon, kinetics.ButlerVolmer):
    """
    Extends :class:`BaseInterfaceLithiumIon` (for exchange-current density, etc) and
    :class:`kinetics.ButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseButlerVolmer(
    BaseInterfaceLithiumIon, inverse_kinetics.InverseButlerVolmer
):
    """
    Extends :class:`BaseInterfaceLithiumIon` (for exchange-current density, etc) and
    :class:`inverse_kinetics.InverseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
