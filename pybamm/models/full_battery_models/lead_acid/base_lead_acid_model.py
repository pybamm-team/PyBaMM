#
# Lead acid base model class
#

import pybamm
import os


class BaseModel(pybamm.BaseBatteryModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lead-acid models


    **Extends:** :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None, name="Unnamed lead-acid model"):
        super().__init__(options, name)
        self.param = pybamm.standard_parameters_lead_acid

    @property
    def default_parameter_values(self):
        input_path = os.path.join(pybamm.root_dir(), "input", "parameters", "lead-acid")
        return pybamm.ParameterValues(
            os.path.join(input_path, "default.csv"),
            {
                "Typical current [A]": 1,
                "Current function": pybamm.GetConstantCurrent(
                    pybamm.standard_parameters_lead_acid.I_typ
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Gu1997.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Gu1997.py"
                ),
                "Electrolyte viscosity": os.path.join(
                    input_path, "electrolyte_viscosity_Chapman1968.py"
                ),
                "Darken thermodynamic factor": os.path.join(
                    input_path, "darken_thermodynamic_factor_Chapman1968.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
                "MacInnes t_plus function": lambda x: 2 * (1 - x),
            },
        )

    @property
    def default_geometry(self):
        if self.options["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")
        elif self.options["dimensionality"] == 2:
            return pybamm.Geometry("2+1D macro")

    def set_standard_output_variables(self):
        super().set_standard_output_variables()
        # Current
        i_cell = pybamm.standard_parameters_lead_acid.current_with_time
        i_cell_dim = (
            pybamm.standard_parameters_lead_acid.dimensional_current_density_with_time
        )
        I = pybamm.standard_parameters_lead_acid.dimensional_current_with_time
        self.variables.update(
            {
                "Total current density": i_cell,
                "Total current density [A.m-2]": i_cell_dim,
                "Current [A]": I,
            }
        )

        # Time
        time_scale = pybamm.standard_parameters_lead_acid.tau_discharge
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
            }
        )

    def set_reactions(self):

        # Should probably refactor as this is a bit clunky at the moment
        # Maybe each reaction as a Reaction class so we can just list names of classes
        param = self.param
        icd = " interfacial current density"
        self.reactions = {
            "main": {
                "Negative": {"s": param.s_n, "aj": "Negative electrode" + icd},
                "Positive": {"s": param.s_p, "aj": "Positive electrode" + icd},
            }
        }
        if "oxygen" in self.options["side reactions"]:
            self.reactions["oxygen"] = {
                "Negative": {
                    "s": -(param.s_plus_Ox + param.t_plus),
                    "s_ox": -param.s_ox_Ox,
                    "aj": "Negative electrode oxygen" + icd,
                },
                "Positive": {
                    "s": -(param.s_plus_Ox + param.t_plus),
                    "s_ox": -param.s_ox_Ox,
                    "aj": "Positive electrode oxygen" + icd,
                },
            }
            self.reactions["main"]["Negative"]["s_ox"] = 0
            self.reactions["main"]["Positive"]["s_ox"] = 0

    def set_soc_variables(self):
        "Set variables relating to the state of charge."
        # State of Charge defined as function of dimensionless electrolyte concentration
        z = pybamm.standard_spatial_vars.z
        soc = (
            pybamm.Integral(self.variables["X-averaged electrolyte concentration"], z)
            * 100
        )
        self.variables.update({"State of Charge": soc, "Depth of Discharge": 100 - soc})

        # Fractional charge input
        if "Fractional Charge Input" not in self.variables:
            fci = pybamm.Variable("Fractional Charge Input", domain="current collector")
            self.variables["Fractional Charge Input"] = fci
            self.rhs[fci] = -self.param.current_with_time * 100
            self.initial_conditions[fci] = self.param.q_init * 100
