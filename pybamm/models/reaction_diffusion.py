#
# Reaction-diffusion model
#
import pybamm


class ReactionDiffusionModel(pybamm.StandardBatteryBaseModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.StandardBatteryBaseModel`

    """

    def __init__(self):
        super().__init__()
        self.variables = {}

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.LeadAcidReaction(param)
        j_n = int_curr_model.get_homogeneous_interfacial_current(["negative electrode"])
        j_p = int_curr_model.get_homogeneous_interfacial_current(["positive electrode"])

        # Porosity
        epsilon = pybamm.Scalar(1)

        # Electrolyte concentration
        j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        reactions = {
            "main": {
                "neg": {"s_plus": 1, "aj": j_n},
                "pos": {"s_plus": 1, "aj": j_p},
                "porosity change": 0,
            }
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, reactions, epsilon)
        self.update(eleclyte_conc_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        c_e_n, _, c_e_p = c_e.orphans
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_parameter_values = (
            pybamm.LeadAcidBaseModel().default_parameter_values
        )
