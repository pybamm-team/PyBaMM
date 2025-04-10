import pybamm
import numpy as np


class TestLumpedThermalModel:
    def test_lumped_thermal_capacity_option(self):
        options = {"thermal": "lumped", "use lumped capacity": "true"}
        model = pybamm.lithium_ion.SPM(options=options)

        model.default_parameter_values

        thermal_submodel = model.submodels["thermal"]
        assert thermal_submodel.use_lumped_capacity == "true"

        options_default = {"thermal": "lumped", "use lumped capacity": "false"}
        model_default = pybamm.lithium_ion.SPM(options=options_default)
        thermal_submodel_default = model_default.submodels["thermal"]
        assert thermal_submodel_default.use_lumped_capacity == "false"

    def test_lumped_thermal_capacity_value(self):
        param = pybamm.ParameterValues("Chen2020")
        lumped_value = 2.5e6
        param["Cell heat capacity [J.K-1.m-3]"] = lumped_value

        options = {"thermal": "lumped", "use lumped capacity": "true"}
        model = pybamm.lithium_ion.SPM(options=options)

        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build()

        var = sim._built_model.variables[
            "Volume-averaged effective heat capacity [J.K-1.m-3]"
        ]

        t_eval = np.array([0])
        y_eval = np.zeros((sim._built_model.len_rhs, 1))
        heat_capacity_eval = var.evaluate(t_eval, y_eval)

        assert abs(heat_capacity_eval - lumped_value) < lumped_value * 1e-10

    def test_fallback_to_component_based(self):
        options = {"thermal": "lumped", "use lumped capacity": "true"}
        model = pybamm.lithium_ion.SPM(options=options)

        sim = pybamm.Simulation(model)
        sim.build()

        assert (
            "Volume-averaged effective heat capacity [J.K-1.m-3]"
            in sim._built_model.variables
        )
