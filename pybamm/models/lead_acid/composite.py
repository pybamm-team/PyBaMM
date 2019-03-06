#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Composite(pybamm.LeadAcidBaseModel):
    """Composite model for lead-acid, from [1].
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    .. math::
        \\frac{\\partial \\tilde{c}}{\\partial t}
        = \\frac{1}{\\varepsilon^{(0)}}\\left(
            \\frac{D^{\\text{eff}, (0)}}{\\mathcal{C}_\\text{e}}
            \\frac{\\partial^2 \\tilde{c}}{\\partial x^2}
            + \\left(
                s + \\beta^{\\text{surf}}c^{(0)}
            \\right)j^{(0)}
        \\right)


    [1] Paper reference

    **Notation for variables and parameters:**

    * f_xy means :math:`f^{(x)}_\\text{y}` (x is the power for the asymptotic \
    expansion and y is the domain). For example c_1n means :math:`c^{(1)}_\\text{n}`, \
    the first-order concentration in the negative electrode
    * fbar_n means :math:`\\bar{f}_n`, the average value of f in that domain, e.g.

    .. math::
        \\text{cbar_n}
        = \\bar{c}_\\text{n}
        = \\frac{1}{\\ell_\\text{n}}
        \\int_0^{\\ell_\\text{n}} \\! c_\\text{n} \\, \\mathrm{d}x

    **Extends:** :class:`pybamm.LeadAcidBaseModel`

    """

    def __init__(self):
        super().__init__()

        # Parameters
        sp = pybamm.standard_parameters
        spla = pybamm.standard_parameters_lead_acid
        # Current function
        i_cell = sp.current_with_time

        # Get leading-order model
        loqs_model = pybamm.lead_acid.LOQS()
        # Get model for the concentration
        # Concentration variables
        c_n = pybamm.Variable(
            "Negative electrode concentration", domain=["negative electrode"]
        )
        c_s = pybamm.Variable("Separator concentration", domain=["separator"])
        c_p = pybamm.Variable(
            "Positive electrode concentration", domain=["positive electrode"]
        )
        c = pybamm.Concatenation(c_n, c_s, c_p)
        # Porosity variables
        eps_n = pybamm.Variable(
            "Negative electrode porosity", domain=["negative electrode"]
        )
        eps_s = pybamm.Variable("Separator porosity", domain=["separator"])
        eps_p = pybamm.Variable(
            "Positive electrode porosity", domain=["positive electrode"]
        )
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)
        # Interfacial current density
        j = pybamm.interface.homogeneous_reaction(i_cell)
        # Concentration model (reaction diffusion with homogeneous reaction)
        conc_model = pybamm.electrolyte.StefanMaxwellDiffusionWithPorosity(
            c, eps, j, spla
        )
        porosity_model = pybamm.electrolyte.Porosity(eps, j)

        # Update own model with submodels
        self.update(loqs_model, conc_model, porosity_model)

        # Extract leading-order variables, taking orphans to remove domains
        c_0 = loqs_model.variables["Concentration"].orphans[0]
        eps_0 = loqs_model.variables["Porosity"]
        eps_0n, eps_0s, eps_0p = [e.orphans[0] for e in eps_0.orphans]
        eta_0n = loqs_model.variables["Negative electrode overpotential"].orphans[0]
        eta_0p = loqs_model.variables["Positive electrode overpotential"].orphans[0]
        Phi_0 = loqs_model.variables["Electrolyte potential"].orphans[0]
        V_0 = loqs_model.variables["Voltage"].orphans[0]

        # Pre-define functions of leading-order variables
        chi_0 = spla.chi(c_0)
        kappa_0n = sp.kappa_e(c_0) * eps_0n ** sp.b
        kappa_0s = sp.kappa_e(c_0) * eps_0s ** sp.b
        kappa_0p = sp.kappa_e(c_0) * eps_0p ** sp.b
        j0_0n = pybamm.Scalar(1)
        j0_0p = pybamm.Scalar(1)
        j0_1n = pybamm.Scalar(1)
        j0_1p = pybamm.Scalar(1)
        dUPbdc = pybamm.Scalar(1)
        dUPbO2dc = pybamm.Scalar(1)

        # Independent variables
        x_n = pybamm.SpatialVariable("x", ["negative electrode"])
        x_s = pybamm.SpatialVariable("x", ["separator"])
        x_p = pybamm.SpatialVariable("x", ["positive electrode"])

        # First-order concentration (c = c0 + C_e * c_1)
        c_1 = (c - c_0) / spla.C_e
        c_1n = (c_n - c_0) / spla.C_e
        c_1p = (c_p - c_0) / spla.C_e

        # Potential
        cbar_1n = pybamm.Scalar(1)  # pybamm.Integral(c_1n, x_n) / sp.l_n
        j0bar_1n = pybamm.Scalar(1)  # pybamm.Integral(j0_1n, x_n) / sp.l_n
        An = (
            j0bar_1n * pybamm.Function(np.tanh, eta_0n) / j0_0n
            - dUPbdc * cbar_1n
            - chi_0 / c_0 * cbar_1n
            + i_cell * sp.l_n / (6 * kappa_0n)
        )

        Phi_1n = -i_cell * x_n ** 2 / (2 * sp.l_n * kappa_0n)
        Phi_1s = -i_cell * ((x_s - sp.l_n) / kappa_0s + sp.l_n / (2 * kappa_0n))
        Phi_1p = -i_cell * (
            sp.l_n / (2 * kappa_0n)
            + sp.l_s / (kappa_0s)
            + (sp.l_p ** 2 - (1 - x_p) ** 2) / (2 * sp.l_p * kappa_0p)
        )
        Phi1 = (
            chi_0 / c_0 * c_1
            + pybamm.Concatenation(
                pybamm.Broadcast(Phi_1n, ["negative electrode"]),
                pybamm.Broadcast(Phi_1s, ["separator"]),
                pybamm.Broadcast(Phi_1p, ["positive electrode"]),
            )
            + An
        )

        # Voltage
        cbar_1p = pybamm.Scalar(1)  # pybamm.Integral(c_1p, x_p) / sp.l_p
        Phibar_1p = pybamm.Scalar(1)  # pybamm.Integral(Phi1, x_p) / sp.l_p
        j0bar_1p = pybamm.Scalar(1)  # pybamm.Integral(j0_1p, x_p) / sp.l_p
        V1 = (
            Phibar_1p
            + dUPbO2dc * cbar_1p
            - j0bar_1p * pybamm.Function(np.tanh, eta_0p) / j0_0p
        )

        # Variables
        self.variables = {
            "c": c,
            "Phi": Phi_0 + spla.C_e * Phi1,
            "V": V_0 + spla.C_e * V1,
        }
