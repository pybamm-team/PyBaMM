#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LOQS(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

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
        epsilon = pybamm.standard_variables.eps_piecewise_constant

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_n, j_p = int_curr_model.get_homogeneous_interfacial_current(broadcast=False)

        # Porosity
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(epsilon, j_n, j_p)

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, j_n, j_p, epsilon)

        self.update(porosity_model, eleclyte_conc_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        j0_n = int_curr_model.get_exchange_current(c_e, domain=["negative electrode"])
        j0_p = int_curr_model.get_exchange_current(c_e, domain=["positive electrode"])

        # Potentials
        ocp_n = param.U_n(c_e)
        ocp_p = param.U_p(c_e)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(
            j_n, j0_n, ["negative electrode"]
        )
        eta_r_p = int_curr_model.get_inverse_butler_volmer(
            j_p, j0_p, ["positive electrode"]
        )
        delta_phi_n = eta_r_n + ocp_n
        delta_phi_p = eta_r_p + ocp_p

        v = delta_phi_n - delta_phi_p
        v_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * v
        self.variables.update({"Terminal voltage": v, "Terminal voltage [V]": v_dim})

        # # Electrolyte current
        # eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        # elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        # self.variables.update(elyte_vars)
        #
        # # Electrode
        # electrode_model = pybamm.electrode.Ohm(param)
        # electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        # self.variables.update(electrode_vars)
