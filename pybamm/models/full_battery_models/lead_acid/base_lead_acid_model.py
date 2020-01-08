#
# Lead acid base model class
#

import pybamm


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
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Sulzer2019)

    @property
    def default_geometry(self):
        if self.options["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")
        elif self.options["dimensionality"] == 2:
            return pybamm.Geometry("2+1D macro")

    @property
    def default_var_pts(self):
        # Choose points that give uniform grid for the standard parameter values
        var = pybamm.standard_spatial_vars
        return {var.x_n: 25, var.x_s: 41, var.x_p: 34, var.y: 10, var.z: 10}

    @property
    def default_solver(self):
        """
        Return default solver based on whether model is ODE model or DAE model.
        There are bugs with KLU on the lead-acid models.
        """
        if len(self.algebraic) == 0:
            return pybamm.ScipySolver()
        elif pybamm.have_scikits_odes():
            return pybamm.ScikitsDaeSolver()
        else:  # pragma: no cover
            return pybamm.CasadiSolver(mode="safe")

    def set_standard_output_variables(self):
        super().set_standard_output_variables()

        # Time
        time_scale = pybamm.standard_parameters_lead_acid.tau_discharge
        self.variables.update(
            {
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
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
        soc = self.variables["X-averaged electrolyte concentration"] * 100
        self.variables.update({"State of Charge": soc, "Depth of Discharge": 100 - soc})

        # Fractional charge input
        if "Fractional Charge Input" not in self.variables:
            fci = pybamm.Variable("Fractional Charge Input", domain="current collector")
            self.variables["Fractional Charge Input"] = fci
            self.rhs[fci] = -self.variables["Total current density"] * 100
            self.initial_conditions[fci] = self.param.q_init * 100
