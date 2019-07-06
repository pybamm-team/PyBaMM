#
# Lithium-ion interface classes
#
import pybamm
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

        if self.domain == "Negative":
            prefactor = 1 / self.param.C_r_n

        elif self.domain == "Positive":
            prefactor = self.param.gamma_p / self.param.C_r_p

        j0 = prefactor * (
            c_e ** (1 / 2) * c_s_surf ** (1 / 2) * (1 - c_s_surf) ** (1 / 2)
        )

        return j0

    def _get_open_circuit_potential(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        # c_s_surf = pybamm.surf(c_s, set_domain=True)

        if self.domain == "Negative":
            ocp = self.param.U_n(c_s_surf)
            dudT = self.param.dUdT_n(c_s_surf)

        elif self.domain == "Positive":
            ocp = self.param.U_p(c_s_surf)
            dudT = self.param.dUdT_p(c_s_surf)

        return ocp, dUdT


class ButlerVolmer(BaseInterfaceLithiumIon, kinetics.BaseButlerVolmer):
    """
    Extends :class:`BaseInterfaceLithiumIon` (for exchange-current density, etc) and
    :class:`kinetics.BaseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)


class InverseButlerVolmer(
    BaseInterfaceLithiumIon, inverse_kinetics.BaseInverseButlerVolmer
):
    """
    Extends :class:`BaseInterfaceLithiumIon` (for exchange-current density, etc) and
    :class:`inverse_kinetics.BaseInverseButlerVolmer` (for kinetics)
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
