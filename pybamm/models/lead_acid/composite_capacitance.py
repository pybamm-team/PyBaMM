#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class CompositeCapacitance(pybamm.LeadAcidBaseModel):
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
        delta_phi_n_av = pybamm.Variable("Average neg electrode potential difference")
        delta_phi_p_av = pybamm.Variable("Average pos electrode potential difference")

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Leading order model
        leading_order_model = pybamm.lead_acid.LOQS()

        # Interfacial current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j_0_n = int_curr_model.get_homogeneous_interfacial_current(neg)
        j_0_p = int_curr_model.get_homogeneous_interfacial_current(pos)
        broad_j_0_n = pybamm.Broadcast(j_0_n, neg)
        broad_j_0_p = pybamm.Broadcast(j_0_p, pos)

        # Average composite interfacial current density
        c_e_n, _, c_e_p = c_e.orphans
        c_e_n_av = pybamm.average(c_e_n)
        c_e_p_av = pybamm.average(c_e_p)
        ocp_n_av = param.U_n(c_e_n_av)
        ocp_p_av = param.U_p(c_e_p_av)
        eta_r_n_av = delta_phi_n_av - ocp_n_av
        eta_r_p_av = delta_phi_p_av - ocp_p_av
        j0_n_av = int_curr_model.get_exchange_current_densities(c_e_n_av, neg)
        j0_p_av = int_curr_model.get_exchange_current_densities(c_e_p_av, pos)
        j_n_av = int_curr_model.get_butler_volmer(j0_n_av, eta_r_n_av, neg)
        j_p_av = int_curr_model.get_butler_volmer(j0_p_av, eta_r_p_av, pos)

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_differential_system(eps, broad_j_0_n, broad_j_0_p)

        # Electrolyte concentration
        reactions = {
            "main": {
                "neg": {"s_plus": param.s_n, "aj": broad_j_0_n},
                "pos": {"s_plus": param.s_p, "aj": broad_j_0_p},
                "porosity change": porosity_model.variables["Porosity change"],
            }
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, reactions, epsilon=eps)

        # Electrolyte current
        reactions_av = {"main": {"neg": {"aj": j_n_av}, "pos": {"aj": j_p_av}}}
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_n.set_leading_order_system(
            delta_phi_n_av, reactions_av, neg
        )
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_p.set_leading_order_system(
            delta_phi_p_av, reactions_av, pos
        )

        self.update(
            leading_order_model,
            porosity_model,
            eleclyte_conc_model,
            eleclyte_current_model_n,
            eleclyte_current_model_p,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(
            j_0_n, j_0_p, j0_n, j0_p
        )
        self.variables.update(j_vars)

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_0_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_0_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte current
        eps0 = leading_order_model.variables["Porosity"]
        c_e_0 = (
            leading_order_model.variables["Electrolyte concentration"]
            .orphans[0]
            .orphans[0]
        )
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_combined(
            ocp_n, eta_r_n, c_e, eps0, c_e_0
        )
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        phi_e = self.variables["Electrolyte potential"]
        electrode_vars = electrode_model.get_explicit_combined(
            ocp_p, eta_r_p, phi_e, eps0
        )
        self.variables.update(electrode_vars)
