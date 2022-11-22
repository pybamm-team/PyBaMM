#
# Tests for the Thevenin equivalant circuit model
#
import pybamm
import unittest


class TestThevenin(unittest.TestCase):
    def test_standard_model(self):
        model = pybamm.equivalent_circuit.Thevenin()
        model.check_well_posedness()

    def test_changing_number_of_rcs(self):
        options = {"number of rc elements": 0}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

        options = {"number of rc elements": 2}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

        options = {"number of rc elements": 3}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

        options = {"number of rc elements": 4}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

        with self.assertRaisesRegex(pybamm.OptionError, "natural numbers"):
            options = {"number of rc elements": -1}
            model = pybamm.equivalent_circuit.Thevenin(options=options)
            model.check_well_posedness()

    def test_calculate_discharge_energy(self):
        options = {"calculate discharge energy": "true"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_power(self):
        options = {"operating mode": "power"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_differential_power(self):
        options = {"operating mode": "differential power"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_resistance(self):
        options = {"operating mode": "resistance"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_differential_resistance(self):
        options = {"operating mode": "differential resistance"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_cccv(self):
        options = {"operating mode": "CCCV"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()

    def test_well_posed_external_circuit_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Terminal voltage [V]"]
            return (
                V
                + I
                - pybamm.FunctionParameter(
                    "Function", {"Time [s]": pybamm.t}, print_name="test_fun"
                )
            )

        options = {"operating mode": external_circuit_function}

        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
