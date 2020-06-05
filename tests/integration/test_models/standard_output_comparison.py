#
# Tests comparing model outputs for standard variables
#
import pybamm
import numpy as np


class StandardOutputComparison(object):
    "Calls all the tests comparing standard output variables."

    def __init__(self, solutions):
        self.solutions = solutions

        if isinstance(solutions[0].model, pybamm.lithium_ion.BaseModel):
            self.chemistry = "Lithium-ion"
        elif isinstance(solutions[0].model, pybamm.lead_acid.BaseModel):
            self.chemistry = "Lead acid"

        self.t = self.get_output_times()

    def get_output_times(self):
        # Get max time allowed from the simulation, i.e. the smallest end time common
        # to all the solutions
        max_t = min([solution.t[-1] for solution in self.solutions])

        # Assign common time
        solution0 = self.solutions[0]
        max_index = np.where(solution0.t == max_t)[0][0]
        t_common = solution0.t[:max_index]

        # Check times
        for solution in self.solutions:
            np.testing.assert_array_equal(t_common, solution.t[:max_index])

        # Get timescale
        timescale = self.solutions[0].model.timescale_eval

        return t_common * timescale

    def run_test_class(self, ClassName, skip_first_timestep=False):
        "Run all tests from a class 'ClassName'"
        if skip_first_timestep:
            t = self.t[1:]
        else:
            t = self.t
        tests = ClassName(t, self.solutions)
        tests.test_all()

    def test_averages(self, skip_first_timestep=False):
        self.run_test_class(AveragesComparison, skip_first_timestep)

    def test_all(self, skip_first_timestep=False):
        self.test_averages(skip_first_timestep)
        self.run_test_class(VariablesComparison, skip_first_timestep)

        if self.chemistry == "Lithium-ion":
            self.run_test_class(ParticleConcentrationComparison, skip_first_timestep)
        elif self.chemistry == "Lead-acid":
            self.run_test_class(PorosityComparison, skip_first_timestep)


class BaseOutputComparison(object):
    def __init__(self, time, solutions):
        self.t = time
        self.solutions = solutions

    def compare(self, var, tol=1e-2):
        "Compare variables from different models"
        # Get variable for each model
        model_variables = [solution[var] for solution in self.solutions]
        var0 = model_variables[0]

        spatial_pts = {}
        if var0.dimensions >= 1:
            spatial_pts[var0.first_dimension] = var0.first_dim_pts
        if var0.dimensions >= 2:
            spatial_pts[var0.second_dimension] = var0.second_dim_pts

        # Calculate tolerance based on the value of var0
        maxvar0 = np.max(abs(var0(self.t, **spatial_pts)))
        if maxvar0 < 1e-14:
            decimal = -int(np.log10(tol))
        else:
            decimal = -int(np.log10(tol * maxvar0))
        # Check outputs are close to each other
        for model_var in model_variables[1:]:
            np.testing.assert_equal(var0.dimensions, model_var.dimensions)
            np.testing.assert_array_almost_equal(
                model_var(self.t, **spatial_pts), var0(self.t, **spatial_pts), decimal
            )


class AveragesComparison(BaseOutputComparison):
    "Compare variables whose average value should be the same across all models"

    def __init__(self, time, solutions):
        super().__init__(time, solutions)

    def test_all(self):
        # Potentials
        self.compare("X-averaged open circuit voltage")
        # Currents
        self.compare("X-averaged negative electrode interfacial current density")
        self.compare("X-averaged positive electrode interfacial current density")
        # Concentration
        self.compare("X-averaged electrolyte concentration")
        # Porosity
        self.compare("X-averaged negative electrode porosity")
        self.compare("X-averaged separator porosity")
        self.compare("X-averaged positive electrode porosity")


class VariablesComparison(BaseOutputComparison):
    "Compare variables across models"

    def __init__(self, time, solutions):
        super().__init__(time, solutions)

    def test_all(self):
        # Concentrations
        self.compare("Electrolyte concentration")
        # self.compare("Reduced cation flux")
        # Potentials
        # Some of these are 'average' but aren't expected to be the same across all
        # models
        self.compare("X-averaged reaction overpotential")
        self.compare("X-averaged negative electrode open circuit potential")
        self.compare("X-averaged positive electrode open circuit potential")
        self.compare("Terminal voltage")
        self.compare("X-averaged solid phase ohmic losses")
        self.compare("Negative electrode reaction overpotential")
        self.compare("Positive electrode reaction overpotential")
        self.compare("Negative electrode potential")
        self.compare("Positive electrode potential")
        self.compare("Electrolyte potential")
        # Currents
        self.compare("Exchange current density")
        self.compare("Negative electrode current density")
        self.compare("Positive electrode current density")


class ParticleConcentrationComparison(BaseOutputComparison):
    def __init__(self, time, solutions):
        super().__init__(time, solutions)

    def test_all(self):
        self.compare("Negative particle concentration")
        self.compare("Positive particle concentration")
        self.compare("Negative particle flux")
        self.compare("Positive particle flux")


class PorosityComparison(BaseOutputComparison):
    def __init__(self, time, solutions):
        super().__init__(time, solutions)

    def test_all(self):
        self.compare("Porosity")
