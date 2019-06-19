#
# Reaction-diffusion model
#
import pybamm


class ReactionDiffusionModel(pybamm.BaseBatteryModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid
        current = param.current_with_time

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.butler_volmer.LeadAcid(param)
        j_n = int_curr_model.get_homogeneous_interfacial_current(current, neg)
        j_p = int_curr_model.get_homogeneous_interfacial_current(current, pos)

        # Porosity
        epsilon = pybamm.Scalar(1)

        # Electrolyte concentration
        j_n = pybamm.Broadcast(j_n, neg)
        j_p = pybamm.Broadcast(j_p, pos)
        self.variables = {"Electrolyte concentration": c_e, "Porosity": epsilon}
        reactions = {
            "main": {
                "neg": {"s_plus": 1, "aj": j_n},
                "pos": {"s_plus": 1, "aj": j_p},
                "porosity change": 0,
            }
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(self.variables, reactions)
        self.update(eleclyte_conc_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

    @property
    def default_parameter_values(self):
        return pybamm.LeadAcidBaseModel().default_parameter_values
