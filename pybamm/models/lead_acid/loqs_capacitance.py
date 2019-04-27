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
        self.variables = {}

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.Variable("Electrolyte concentration")
        delta_phi_n = pybamm.Variable("Negative electrode potential difference")
        delta_phi_p = pybamm.Variable("Positive electrode potential difference")
        epsilon = pybamm.standard_variables.eps_piecewise_constant

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Exchange-current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j0_n = int_curr_model.get_exchange_current(c_e, domain=["negative electrode"])
        j0_p = int_curr_model.get_exchange_current(c_e, domain=["positive electrode"])

        # Open-circuit potential and reaction overpotential
        eta_r_n = delta_phi_n - param.U_n(c_e)
        eta_r_p = delta_phi_p - param.U_p(c_e)

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n, ["negative electrode"])
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p, ["positive electrode"])

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(epsilon, j_n, j_p)

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, j_n, j_p, epsilon)

        # Electrolyte current
        eleclyte_current_model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_n.set_leading_order_system(
            delta_phi_n, j_n, ["negative electrode"]
        )
        eleclyte_current_model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        eleclyte_current_model_p.set_leading_order_system(
            delta_phi_p, j_p, ["positive electrode"]
        )
        self.update(
            porosity_model,
            eleclyte_conc_model,
            eleclyte_current_model_n,
            eleclyte_current_model_p,
        )

        v = delta_phi_n - delta_phi_p
        v_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * v
        self.variables.update({"Terminal voltage [V]": v_dim})

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        for name, var in leading_order_variables.items():
            if "negative" in name.lower():
                self.variables[name] = pybamm.Broadcast(var, ["negative electrode"])
            elif "positive" in name.lower():
                self.variables[name] = pybamm.Broadcast(var, ["positive electrode"])
            # else:
            #     self.variables[name] = var

        # self.variables.update(
        #     {
        #         "Interfacial current density": pybamm.Scalar(1),
        #         "Interfacial current density [A m-2]": pybamm.Scalar(1),
        #         "Exchange-current density": pybamm.Scalar(1),
        #         "Exchange-current density [A m-2]": pybamm.Scalar(1),
        #     }
        # )
        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Default Solver"

        # Use stiff solver
        self.default_solver = pybamm.ScipySolver("BDF")
