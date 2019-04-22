#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Composite(pybamm.LeadAcidBaseModel):
    """Composite model for lead-acid, from [1]_.
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
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    """

    def __init__(self):
        # Update own model with submodels
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Leading-order model
        loqs_model = pybamm.lead_acid.LOQS()
        self.update(loqs_model)
        # Label variables as leading-order
        self.variables = {
            name + " (leading-order)": var for name, var in self.variables.items()
        }

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_vars = int_curr_model.get_homogeneous_interfacial_current()
        self.variables.update(j_vars)

        # Porosity
        j = j_vars["Interfacial current density"]
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_differential_system(eps, j)
        self.update(porosity_model)

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, self.variables)
        self.update(eleclyte_conc_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        ecd_vars = int_curr_model.get_exchange_current_densities(
            self.variables, intercalation=False
        )
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        c_e_n = self.variables["Negative electrode electrolyte concentration"]
        c_e_p = self.variables["Positive electrode electrolyte concentration"]
        ocp_vars = pot_model.get_open_circuit_potentials(c_e_n, c_e_p)
        eta_r_vars = pot_model.get_reaction_overpotentials(self.variables, "current")
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_combined(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)
