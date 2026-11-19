#
# Coupled ion-electron transfer kinetics class
#
from __future__ import annotations

import numpy as np

import pybamm

from .base_kinetics import BaseKinetics


class CoupledIonElectronTransfer(BaseKinetics):
    """
    Intercalation rate law in which the ion and electron transfer in one concerted
    step :footcite:t:`Fraggedakis2021`, using the asymptotic Marcus-Hush-Chidsey
    integral of :footcite:t:`Zeng2014`. In units of :math:`RT/F`,

    .. math::
        j = k_0 \\sqrt{\\pi\\lambda}\\,\\mathrm{erfc}\\left(\\frac{\\lambda - \\sqrt{1 + \\sqrt{\\lambda} + \\eta_f^2}}{2\\sqrt{\\lambda}}\\right) (1 - \\theta) \\left[\\frac{\\theta}{1 + e^{-\\eta_f}} - \\frac{\\tilde{c}_e}{1 + e^{\\eta_f}}\\right],

    where :math:`\\eta_f = \\eta_r + \\ln(\\tilde{c}_e / \\theta)`,
    :math:`\\theta = c_{s,surf}/c_{max}` and :math:`\\tilde{c}_e = c_e/c_{e,init}`.
    Not available for a lithium metal electrode, which has no host lattice.

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
        if reaction == "lithium metal plating":
            raise pybamm.OptionError(
                "Coupled ion-electron transfer kinetics need a host lattice, so "
                "they cannot be used for the lithium metal electrode of a half-cell. "
                "Set 'intercalation kinetics' per electrode instead, e.g. "
                "('Marcus-Hush-Chidsey', 'coupled ion-electron transfer') for a "
                "positive working electrode"
            )
        if reaction != "lithium-ion main":
            raise pybamm.OptionError(
                "Coupled ion-electron transfer kinetics are only implemented for "
                f"lithium-ion intercalation, not '{reaction}'"
            )
        super().__init__(param, domain, reaction, options, phase)
        domain_options = getattr(self.options, self.domain)
        if getattr(domain_options, self.phase)["exchange-current density"] != "single":
            raise pybamm.OptionError(
                "Coupled ion-electron transfer kinetics do not use the "
                "exchange-current density parameter, so 'exchange-current density' "
                "must be 'single'"
            )
        pybamm.citations.register("Fraggedakis2021")
        pybamm.citations.register("Zeng2014")

    def get_coupled_variables(self, variables):
        # The rate law needs the concentrations, but BaseKinetics only passes j0
        # and the overpotential to _get_kinetics
        self._concentrations = self._get_surface_fractions(variables)
        return super().get_coupled_variables(variables)

    def _get_surface_fractions(self, variables):
        """Return the surface stoichiometry and relative electrolyte concentration,
        floored so the logarithm in the formal overpotential stays finite."""
        c_e, c_s_surf, _ = self._get_intercalation_concentrations(variables)
        tol = pybamm.settings.tolerances["ciet__c"]
        stoichiometry = pybamm.maximum(
            pybamm.minimum(c_s_surf / self.phase_param.c_max, 1 - tol), tol
        )
        relative_c_e = pybamm.maximum(c_e / self.phase_param.c_e_init, tol)
        return stoichiometry, relative_c_e

    def _get_exchange_current_density(self, variables):
        stoichiometry, relative_c_e = self._get_surface_fractions(variables)
        _, _, T = self._get_intercalation_concentrations(variables)

        # Linear-response slope, so j ~ j0 F eta_r / RT as for one-electron
        # Butler-Volmer
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
        return pybamm.log(relative_c_e / stoichiometry)

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

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        stoichiometry, relative_c_e = (
            self._broadcast_to_eta_r_domain(c, eta_r) for c in self._concentrations
        )

        F_RT = self.param.F / (self.param.R * T)
        eta_f = F_RT * eta_r + self._formal_shift(stoichiometry, relative_c_e)
        oxidation = stoichiometry * self._fermi(eta_f)
        reduction = relative_c_e * self._fermi(-eta_f)
        # The transition state excludes a lattice site, so both directions need one
        return (
            u
            * self._rate_factor(eta_f, F_RT, T)
            * (1 - stoichiometry)
            * (oxidation - reduction)
        )
