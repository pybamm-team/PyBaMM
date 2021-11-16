#
# Tests for the lithium-ion SPM model
#
import pybamm
import unittest


class TestSPM(unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.SPM(options)

    def test_new_model(self):
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"})
        new_model = model.new_copy()
        model_T_eqn = model.rhs[model.variables["Cell temperature"]]
        new_model_T_eqn = new_model.rhs[new_model.variables["Cell temperature"]]
        self.assertEqual(new_model_T_eqn.id, model_T_eqn.id)
        self.assertEqual(new_model.name, model.name)
        self.assertEqual(new_model.use_jacobian, model.use_jacobian)
        self.assertEqual(new_model.convert_to_format, model.convert_to_format)
        self.assertEqual(new_model.timescale.id, model.timescale.id)

        # with custom submodels
        options = {"stress-induced diffusion": "false", "thermal": "x-full"}
        model = pybamm.lithium_ion.SPM(options, build=False)
        particle_n = pybamm.particle.no_distribution.XAveragedPolynomialProfile(
            model.param, "Negative", "quadratic profile", options
        )
        model.submodels["negative particle"] = particle_n
        model.build_model()
        new_model = model.new_copy()
        new_model_cs_eqn = list(new_model.rhs.values())[1]
        model_cs_eqn = list(model.rhs.values())[1]
        self.assertEqual(new_model_cs_eqn.id, model_cs_eqn.id)


class TestSPMExternalCircuits(unittest.TestCase):
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_cccv(self):
        options = {"operating mode": "CCCV"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Terminal voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

        options = {"operating mode": external_circuit_function}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_stress_induced_diffusion_mixed(self):
        options = {
            "particle mechanics": "swelling only",
            "stress-induced diffusion": ("true", "false"),
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
