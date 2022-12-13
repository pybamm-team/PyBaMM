#
# Tests for inputting a temperature profile
#
import pybamm
import unittest
import numpy as np


class TestInputLumpedTemperature(unittest.TestCase):
    def test_input_lumped_temperature(self):
        model = pybamm.lithium_ion.SPMe()
        parameter_values = model.default_parameter_values
        # in the default isothermal model, the temperature is everywhere equal
        # to the ambient temperature
        parameter_values["Ambient temperature [K]"] = pybamm.InputParameter(
            "Volume-averaged cell temperature [K]"
        )
        sim = pybamm.Simulation(model)

        t_eval = np.linspace(0, 100, 3)

        T_av = 298

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            inputs = {"Volume-averaged cell temperature [K]": T_av}
            T_av += 1
            sim.step(dt, inputs=inputs)  # works


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
