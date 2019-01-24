#
# Standard basic tests for any model
#
import pybamm


class StandardModelTest(object):
    def __init__(self, model):
        self.model = model
        self.param = None
        self.mesh = None

    def test_processing_parameters(self, param_str="LCO"):
        if param_str == "LCO":
            self.param = pybamm.ParameterValues(
                "input/parameters/lithium-ion/parameters/LCO.csv"
            )

        self.param.process_model(self.model)
        # Model should still be well-posed after processing
        # self.model.check_well_posedness()

    def test_processing_disc(self, disc_str="Finite Volume"):
        if disc_str == "Finite Volume":
            self.mesh = pybamm.FiniteVolumeMacroMesh(self.param, 2)
            disc = pybamm.FiniteVolumeDiscretisation(self.mesh)
        disc.process_model(self.model)
        # Model should still be well-posed after processing
        # self.model.check_well_posedness()

    def test_solving(self):
        solver = pybamm.ScipySolver(tol=1e-8, method="RK45")
        t_eval = self.mesh["time"]
        solver.solve(self.model, t_eval)

    def test_all(self):
        self.test_processing_parameters()
        self.test_processing_disc()
        self.test_solving()
