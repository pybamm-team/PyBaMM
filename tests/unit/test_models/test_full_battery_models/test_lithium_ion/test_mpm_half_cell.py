#
# Tests for the lithium-ion MPM model
#
from tests import TestCase
import pybamm
import unittest


class TestMPM(TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal", "working electrode": "positive"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

        # Test build after init
        model = pybamm.lithium_ion.MPM({"working electrode": "positive"}, build=False)
        model.build_model()
        model.check_well_posedness()

    def test_default_parameter_values(self):
        # check default parameters are added correctly
        model = pybamm.lithium_ion.MPM({"working electrode": "positive"})
        self.assertEqual(
            model.default_parameter_values[
                "Positive area-weighted mean particle radius [m]"
            ],
            5.3e-06,
        )

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped", "working electrode": "positive"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile", "working electrode": "positive"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_differential_surface_form(self):
        options = {
            "surface form": "differential",
            "working electrode": "positive",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMExternalCircuits(TestCase):
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage", "working electrode": "positive"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power", "working electrode": "positive"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

        options = {
            "operating mode": external_circuit_function,
            "working electrode": "positive",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
