#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LOQSCapacitance(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid, with capacitance effects

    **Extends**: :class:`pybamm.LeadAcidBaseModel`

    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.Variable("Electrolyte concentration")
        delta_phi_n = pybamm.Variable("Negative electrode potential difference")
        delta_phi_p = pybamm.Variable("Positive electrode potential difference")
        eps = pybamm.standard_variables.eps_piecewise_constant
        leading_order_variables = {
            "Electrolyte concentration": c_e,
            "Negative electrode potential difference": delta_phi_n,
            "Positive electrode potential difference": delta_phi_p,
        }

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Exchange-current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        leading_order_ecd_vars = int_curr_model.get_exchange_current_densities(
            leading_order_variables, intercalation=False
        )
        leading_order_variables.update(leading_order_ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        leading_order_ocp_vars = pot_model.get_open_circuit_potentials(c_e, c_e)
        leading_order_variables.update(leading_order_ocp_vars)
        leading_order_eta_r_vars = pot_model.get_reaction_overpotentials(
            leading_order_variables, "potential differences"
        )
        leading_order_variables.update(leading_order_eta_r_vars)

        # Interfacial current density
        leading_order_j_vars = int_curr_model.get_interfacial_current_butler_volmer(
            leading_order_variables
        )
        leading_order_variables.update(leading_order_j_vars)

        # Porosity
        j = leading_order_j_vars["Interfacial current density"]
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(eps, j)
        self.update(porosity_model)

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, leading_order_variables)
        self.update(eleclyte_conc_model)

        # Electrolyte current
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_n.set_leading_order_system(
            delta_phi_n, leading_order_variables, ["negative electrode"]
        )
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_p.set_leading_order_system(
            delta_phi_p, leading_order_variables, ["positive electrode"]
        )
        self.update(eleclyte_current_model_n, eleclyte_current_model_p)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)
