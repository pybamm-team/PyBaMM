#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


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
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lead_acid.__dict__)

        #
        # Variables
        #
        # Concentration variables
        c_e_n = pybamm.Variable(
            "Negative electrode concentration", domain=["negative electrode"]
        )
        c_e_s = pybamm.Variable("Separator concentration", domain=["separator"])
        c_e_p = pybamm.Variable(
            "Positive electrode concentration", domain=["positive electrode"]
        )
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        # Porosity variables
        eps_n = pybamm.Variable(
            "Negative electrode porosity", domain=["negative electrode"]
        )
        eps_s = pybamm.Variable("Separator porosity", domain=["separator"])
        eps_p = pybamm.Variable(
            "Positive electrode porosity", domain=["positive electrode"]
        )
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        #
        # Submodels
        #
        # Leading-order model
        loqs_model = pybamm.lead_acid.LOQS()
        # Interfacial current density
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        j = pybamm.interface.homogeneous_reaction(whole_cell)
        # Concentration model (reaction diffusion with homogeneous reaction)
        conc_model = pybamm.electrolyte_diffusion.StefanMaxwellWithPorosity(
            c_e, eps, j, param
        )
        # Porosity model
        porosity_model = pybamm.porosity.Standard(eps, j, param)
        # Electrolyte potential model (solve ODE analytically)
        electrolyte_potential_model = pybamm.electrolyte_current.FirstOrderPotential(
            loqs_model, c_e, param
        )

        # Update own model with submodels
        self.update(loqs_model, conc_model, porosity_model, electrolyte_potential_model)
