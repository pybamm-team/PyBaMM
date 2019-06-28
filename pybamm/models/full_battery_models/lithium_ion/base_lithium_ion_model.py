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

    def __init__(self, options=None):
        super().__init__(options)
        self.param = pybamm.standard_parameters_lithium_ion

    def set_standard_output_variables(self):
        super().set_standard_output_variables()
        # Additional standard output variables
        # Time
        time_scale = pybamm.standard_parameters_lithium_ion.tau_discharge
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
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
