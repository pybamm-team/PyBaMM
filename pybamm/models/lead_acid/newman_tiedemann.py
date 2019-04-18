#
# Lead-acid Newman-Tiedemann model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class NewmanTiedemann(pybamm.LeadAcidBaseModel):
    """Porous electrode model for lead-acid, from [1]_.

    .. math::
        \\frac{\\partial \\tilde{c}}{\\partial t}
        = \\frac{1}{\\varepsilon^{(0)}}\\left(
            \\frac{D^{\\text{eff}, (0)}}{\\mathcal{C}_\\text{e}}
            \\frac{\\partial^2 \\tilde{c}}{\\partial x^2}
            + \\left(
                s + \\beta^{\\text{surf}}c^{(0)}
            \\right)j^{(0)}
        \\right)


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

    References
    ==========
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: I. Physical Model.
           arXiv preprint arXiv:1902.01771, 2019.

    """

    def __init__(self):
        super().__init__()

        # Parameters
        param = pybamm.standard_parameters_lead_acid

        #
        # Variables
        #
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        #
        # Variables
        #
        # Concentration variables
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", domain=["negative electrode"]
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain=["separator"]
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain=["positive electrode"]
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
        # Potential variables
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain=["negative electrode"]
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential", domain=["separator"]
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain=["positive electrode"]
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", domain=["negative electrode"]
        )
        phi_s_s = pybamm.Broadcast(pybamm.Scalar(0), ["separator"])
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", domain=["positive electrode"]
        )
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)
        Delta_phi = pybamm.Concatenation(
            phi_s_n - phi_e_n, phi_s_s - phi_e_s, phi_s_p - phi_e_p
        )
        #
        # Submodels
        #
        # Interfacial current density
        j = pybamm.interface.butler_volmer(param, c_e, Delta_phi, domain=whole_cell)
        # Concentration model (reaction diffusion with butler volmer)
        conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            c_e, j, param, epsilon=eps
        )
        # Porosity model
        porosity_model = pybamm.porosity.Standard(eps, j, param)
        # Electrolyte potential model (conservation of current and MacInnes)
        electrolyte_potential_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            c_e, phi_e, j, param, eps=eps
        )
        # Solid potential model (conservation of current and MacInnes)
        solid_potential_model = pybamm.electrode.Ohm(phi_s, j, param, eps=eps)

        # Update own model with submodels
        self.update(
            conc_model,
            porosity_model,
            electrolyte_potential_model,
            solid_potential_model,
        )

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
