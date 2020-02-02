#
# Lithium-ion base model class
#
import pybamm


class BaseModel(pybamm.BaseBatteryModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lithium-ion models

    **Extends:** :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None, name="Unnamed lithium-ion model"):
        super().__init__(options, name)
        self.param = pybamm.standard_parameters_lithium_ion

    def set_standard_output_variables(self):
        super().set_standard_output_variables()

        # Time
        time_scale = pybamm.standard_parameters_lithium_ion.tau_discharge
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
            }
        )

        # Particle concentration and position
        self.variables.update(
            {
                "Negative particle concentration": None,
                "Positive particle concentration": None,
                "Negative particle surface concentration": None,
                "Positive particle surface concentration": None,
                "Negative particle concentration [mol.m-3]": None,
                "Positive particle concentration [mol.m-3]": None,
                "Negative particle surface concentration [mol.m-3]": None,
                "Positive particle surface concentration [mol.m-3]": None,
            }
        )
        var = pybamm.standard_spatial_vars
        param = pybamm.geometric_parameters
        self.variables.update(
            {
                "r_n": var.r_n,
                "r_n [m]": var.r_n * param.R_n,
                "r_p": var.r_p,
                "r_p [m]": var.r_p * param.R_p,
            }
        )

    def set_reactions(self):

        # Should probably refactor as this is a bit clunky at the moment
        # Maybe each reaction as a Reaction class so we can just list names of classes
        icd = " interfacial current density"
        self.reactions = {
            "main": {
                "Negative": {
                    "s": 1 - self.param.t_plus,
                    "aj": "Negative electrode" + icd,
                },
                "Positive": {
                    "s": 1 - self.param.t_plus,
                    "aj": "Positive electrode" + icd,
                },
            }
        }
