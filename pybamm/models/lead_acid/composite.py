#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Composite(pybamm.BaseModel):
    """Composite model for lead-acid, from [1].

    .. math::
        \\frac{\\partial \\tilde{c}}{\\partial t} =
        \\frac{1}{\\varepsilon^{(0)}}

    [1] Paper reference

    Notation for variables and parameters:

    **Extends:** :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()

        # Get leading-order model
        loqs_model = pybamm.lead_acid.LOQS()
        # Get model for the concentration (reaction diffusion with homogeneous reaction)
        j = pybamm.interface.homogeneous_reaction()
        conc_model = pybamm.electrolyte.StefanMaxwellDiffusionWithPorosity(j)

        # Update own model with submodels
        self.update(loqs_model, conc_model)

        # Extract leading-order variables
        c0 = self.variables["c0"]  # concentration
        eps0 = self.variables["eps0"]  # porosities
        eps_0n, eps_0s, eps_0p = eps0.children
        eta_0n = self.variables["eta_0n"]  # overpotentials
        eta_0p = self.variables["eta_0p"]
        Phi0 = self.variables["Phi0"]
        V0 = self.variables["V0"]
        # Extract composite variables
        c = self.variables["c"]  # composite concentration, c = c0 + Cd * c1

        # Independent variables
        xn = pybamm.Space("negative electrode")
        xs = pybamm.Space("separator")
        xp = pybamm.Space("positive electrode")

        # Parameters
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls
        lp = pybamm.standard_parameters.lp
        Cd = pybamm.standard_parameters_lead_acid.Cd
        # Functions
        icell = pybamm.standard_parameters_lead_acid.icell(pybamm.t)
        kappa_0n = pybamm.standard_parameters_lead_acid.kappa(c0, eps_0n)
        kappa_0s = pybamm.standard_parameters_lead_acid.kappa(c0, eps_0s)
        kappa_0p = pybamm.standard_parameters_lead_acid.kappa(c0, eps_0p)
        chi0 = pybamm.standard_parameters_lead_acid.chi(c0)
        j0_0n = pybamm.standard_parameters_lead_acid.j0n(c0)
        j0_0p = pybamm.standard_parameters_lead_acid.j0p(c0)
        j0_1n = 1
        j0_1p = 1
        dUPbdc = pybamm.standard_parameters_lead_acid.dUPbdc
        dUPbO2dc = pybamm.standard_parameters_lead_acid.dUPbO2dc

        # First-order concentration (c = c0 + Cd * c1)
        c1 = (c - c0) / Cd

        # Potential
        cbar_1n = pybamm.Integral(c1, xn) / ln
        j0bar_1n = pybamm.Integral(j0_1n, xn) / ln
        An = (
            j0bar_1n * pybamm.Function(eta_0n, np.tanh) / j0_0n
            - dUPbdc(c0) * cbar_1n
            - chi0 / c0 * cbar_1n
            + icell * ln / (6 * kappa_0n)
        )

        Phi_1n = -icell * xn ** 2 / (2 * ln * kappa_0n)
        Phi_1s = -icell * ((xs - ln) / kappa_0s + ln / (2 * kappa_0n))
        Phi_1p = -icell * (
            ln / (2 * kappa_0n)
            + ls / (kappa_0s)
            + (lp ** 2 - (1 - xp) ** 2) / (2 * lp * kappa_0p)
        )
        Phi1 = chi0 / c0 * c1 + pybamm.Concatenation(Phi_1n, Phi_1s, Phi_1p) + An

        # Voltage
        cbar_1p = pybamm.Integral(c1, xp) / lp
        Phibar_1p = pybamm.Integral(Phi1, xp) / lp
        j0bar_1p = pybamm.Integral(j0_1p, xp) / lp
        V1 = (
            Phibar_1p
            + dUPbO2dc(c0) * cbar_1p
            - j0bar_1p * pybamm.Function(eta_0p, np.tanh) / j0_0p
        )

        # Variables
        self.variables = {"c": c, "Phi": Phi0 + Cd * Phi1, "V": V0 + Cd * V1}

        # Overwrite default parameter values
        self.default_parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"current scale": 1}
        )
