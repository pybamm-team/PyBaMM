#
# Coupled ion-electron transfer kinetics class
#
from __future__ import annotations

import numpy as np

import pybamm

from .base_kinetics import BaseKinetics
from .marcus import MarcusHushChidsey


class CoupledIonElectronTransfer(MarcusHushChidsey):
    """
    Intercalation rate law in which the ion and electron transfer in one concerted
    step :footcite:t:`Fraggedakis2021`, using the asymptotic Marcus-Hush-Chidsey
    integral of :footcite:t:`Zeng2014`. In units of :math:`RT/F`,

    .. math::
        j = k_0 \\sqrt{\\pi\\lambda}\\,\\mathrm{erfc}\\left(\\frac{\\lambda - \\sqrt{1 + \\sqrt{\\lambda} + \\eta_f^2}}{2\\sqrt{\\lambda}}\\right) \\left[\\frac{\\theta}{1 + e^{-\\eta_f}} - \\frac{\\tilde{c}_e (1 - \\theta)}{1 + e^{\\eta_f}}\\right],

    where :math:`\\eta_f = \\eta_r + \\ln(\\tilde{c}_e (1 - \\theta) / \\theta)`,
    :math:`\\theta = c_{s,surf}/c_{max}` and :math:`\\tilde{c}_e = c_e/c_{e,init}`.
    Lithium metal plating falls back to Marcus-Hush-Chidsey kinetics.

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
        if reaction not in ["lithium-ion main", "lithium metal plating"]:
            raise pybamm.OptionError(
                "Coupled ion-electron transfer kinetics are only implemented for "
                f"lithium-ion intercalation, not '{reaction}'"
            )
        # Skip MarcusHushChidsey.__init__ so Sripad2020 is only cited for plating
        BaseKinetics.__init__(self, param, domain, reaction, options, phase)
        if reaction == "lithium metal plating":
            pybamm.citations.register("Sripad2020")
            return
        domain_options = getattr(self.options, self.domain)
        if getattr(domain_options, self.phase)["exchange-current density"] != "single":
            raise pybamm.OptionError(
                "Coupled ion-electron transfer kinetics do not use the "
                "exchange-current density parameter, so 'exchange-current density' "
                "must be 'single'"
            )
        pybamm.citations.register("Fraggedakis2021")
        pybamm.citations.register("Zeng2014")

    def _get_exchange_current_density(self, variables):
        if self.reaction != "lithium-ion main":
            return super()._get_exchange_current_density(variables)

        c_e, c_s_surf, T = self._get_intercalation_concentrations(variables)
        phase_param = self.phase_param
        tol_c_s = pybamm.settings.tolerances["j0__c_s"]
        tol_c_e = pybamm.settings.tolerances["j0__c_e"]
        stoichiometry = pybamm.maximum(
            pybamm.minimum(c_s_surf / phase_param.c_max, 1 - tol_c_s), tol_c_s
        )
        relative_c_e = pybamm.maximum(c_e, tol_c_e) / phase_param.c_e_init

        # Kept for the rate law, which only receives j0 and the overpotential
        self._stoichiometry = stoichiometry
        self._relative_c_e = relative_c_e

        # Linear-response slope, so j ~ j0 F eta_r / RT as for Butler-Volmer
        F_RT = self.param.F / (self.param.R * T)
        eta_f_equilibrium = self._formal_shift(stoichiometry, relative_c_e)
        return (
            relative_c_e
            * (1 - stoichiometry)
            * self._rate_factor(eta_f_equilibrium, F_RT, T)
            * self._fermi(-eta_f_equilibrium)
        )

    @staticmethod
    def _formal_shift(stoichiometry, relative_c_e):
        return pybamm.log(relative_c_e * (1 - stoichiometry) / stoichiometry)

    @staticmethod
    def _fermi(x):
        """Return 1 / (1 + exp(-x)) without overflow."""
        return (1 + pybamm.tanh(x / 2)) / 2

    def _rate_factor(self, eta_f, F_RT, T):
        """Return k0 times the Fermi-weighted Marcus integral prefactor."""
        lambda_T = F_RT * self.phase_param.mhc_lambda
        k0 = self.phase_param.k0_coupled_ion_electron_transfer(T)
        a = 1 + pybamm.sqrt(lambda_T)
        arg = (lambda_T - pybamm.sqrt(a + eta_f**2)) / (2 * pybamm.sqrt(lambda_T))
        return k0 * pybamm.sqrt(np.pi * lambda_T) * pybamm.erfc(arg)

    def _match_domains(self, symbol, eta_r):
        domain = self.domain
        if symbol.secondary_domain == [
            "current collector"
        ] and eta_r.secondary_domain == [f"{domain} electrode"]:
            symbol = pybamm.SecondaryBroadcast(symbol, [f"{domain} electrode"])
        return symbol

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        if self.reaction != "lithium-ion main":
            return super()._get_kinetics(j0, ne, eta_r, T, u)

        stoichiometry = self._match_domains(self._stoichiometry, eta_r)
        relative_c_e = self._match_domains(self._relative_c_e, eta_r)

        F_RT = self.param.F / (self.param.R * T)
        eta_f = F_RT * eta_r + self._formal_shift(stoichiometry, relative_c_e)
        oxidation = stoichiometry * self._fermi(eta_f)
        reduction = relative_c_e * (1 - stoichiometry) * self._fermi(-eta_f)
        return u * self._rate_factor(eta_f, F_RT, T) * (oxidation - reduction)
