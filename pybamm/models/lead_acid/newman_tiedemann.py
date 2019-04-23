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

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e
        eps = pybamm.standard_variables.eps
        phi_e = pybamm.standard_variables.phi_e
        phi_s_p = pybamm.standard_variables.phi_s_p
        phi_s_n = pybamm.standard_variables.phi_s_n

        # Add variables to list of variables, as they are needed by submodels
        self.variables.update(
            {
                "Electrolyte concentration": c_e,
                "Porosity": eps,
                "Electrolyte potential": phi_e,
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
            }
        )

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Exchange-current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        ecd_vars = int_curr_model.get_exchange_current_densities(
            self.variables, intercalation=False
        )
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        c_e_n = self.variables["Electrolyte concentration"].orphans[0]
        c_e_p = self.variables["Electrolyte concentration"].orphans[2]
        ocp_vars = pot_model.get_open_circuit_potentials(c_e_n, c_e_p)
        self.variables.update(ocp_vars)
        eta_r_vars = pot_model.get_reaction_overpotentials(self.variables, "potentials")
        self.variables.update(eta_r_vars)

        # Interfacial current density
        j_vars = int_curr_model.get_interfacial_current_butler_volmer(self.variables)
        self.variables.update(j_vars)

        # Porosity
        j = j_vars["Interfacial current density"]
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_differential_system(eps, j)
        self.update(porosity_model)

        # Electrolyte diffusion
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(c_e, self.variables)

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_algebraic_system(phi_e, self.variables)

        # Electrode
        negative_electrode_current_model = pybamm.electrode.Ohm(param)
        negative_electrode_current_model.set_algebraic_system(phi_s_n, self.variables)
        positive_electrode_current_model = pybamm.electrode.Ohm(param)
        positive_electrode_current_model.set_algebraic_system(phi_s_p, self.variables)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"
        self.update(
            electrolyte_diffusion_model,
            eleclyte_current_model,
            negative_electrode_current_model,
            positive_electrode_current_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-process"
        volt_vars = positive_electrode_current_model.get_post_processed(self.variables)
        self.variables.update(volt_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # Default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
