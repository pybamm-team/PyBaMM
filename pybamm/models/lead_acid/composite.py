#
# Lead-acid Composite model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Composite(pybamm.LeadAcidBaseModel):
    """Composite model for lead-acid, from [1]_.
    Uses leading-order model from :class:`pybamm.lead_acid.LOQS`

    References
    ==========
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends:** :class:`pybamm.LeadAcidBaseModel`
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
        electrolyte_concentration_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            param
        )
        electrolyte_concentration_model.set_differential_system(
            c_e, reactions, epsilon=eps
        )

        self.update(
            leading_order_model, porosity_model, electrolyte_concentration_model
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        j_0_vars = int_curr_model.get_derived_interfacial_currents(
            j_0_n, j_0_p, j0_n, j0_p
        )
        self.variables.update(j_0_vars)

        # Potentials
        ocp_n = param.U_n(c_e_n)
        ocp_p = param.U_p(c_e_p)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_0_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_0_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        eps0 = leading_order_model.variables["Porosity"]
        c_e_0 = (
            leading_order_model.variables["Electrolyte concentration"]
            .orphans[0]
            .orphans[0]
        )

        # Load electrolyte and electrode potentials
        electrode_model = pybamm.electrode.Ohm(param)
        electrolyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            param
        )

        # Negative electrode potential
        phi_s_n = electrode_model.get_neg_pot_explicit_combined(eps0)

        # Electrolyte potential
        electrolyte_vars = electrolyte_current_model.get_explicit_combined(
            ocp_n, eta_r_n, c_e, phi_s_n, eps0, c_e_0
        )
        phi_e = self.variables["Electrolyte potential"]
        self.variables.update(electrolyte_vars)

        # Electrode
        electrode_vars = electrode_model.get_explicit_combined(
            phi_s_n, phi_e, ocp_p, eta_r_p, eps0
        )
        self.variables.update(electrode_vars)
