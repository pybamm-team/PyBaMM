import pybamm
import numpy as np


class TestLumpedThermalModel:
    def test_lumped_thermal_capacity_option(self):
        options = {"thermal": "lumped", "use lumped thermal capacity": "true"}
        model = pybamm.lithium_ion.SPM(options=options)

        model.default_parameter_values

        thermal_submodel = model.submodels["thermal"]
        assert thermal_submodel.use_lumped_thermal_capacity == "true"

        options_default = {"thermal": "lumped", "use lumped thermal capacity": "false"}
        model_default = pybamm.lithium_ion.SPM(options=options_default)
        thermal_submodel_default = model_default.submodels["thermal"]
        assert thermal_submodel_default.use_lumped_thermal_capacity == "false"

    def test_lumped_thermal_capacity(self):
        options = {"thermal": "lumped", "use lumped thermal capacity": "true"}
        model = pybamm.lithium_ion.SPM(options=options)

        param = pybamm.ParameterValues("Chen2020")
        param.update(
            {
                "Cell heat capacity [J.K-1.m-3]": 2.5e6,
            },
            check_already_exists=False,
        )

        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build()

        var = sim._built_model.variables[
            "Volume-averaged effective heat capacity [J.K-1.m-3]"
        ]

        t_eval = np.array([0])
        y_eval = np.zeros((sim._built_model.len_rhs, 1))
        heat_capacity = var.evaluate(t_eval, y_eval)

        np.testing.assert_allclose(heat_capacity, 2.5e6, rtol=1e-10)
