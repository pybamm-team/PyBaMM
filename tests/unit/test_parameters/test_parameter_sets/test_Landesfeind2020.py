#
# Tests for LG M50 parameter set loads
#
import pybamm
import unittest
import os
import numpy as np


class TestLandesfeind(unittest.TestCase):
    def test_electrolyte_conductivity(self):
        root = pybamm.root_dir()
        p = "pybamm/input/parameters/lithium-ion/electrolytes/lipf6_Landesfeind2019"
        k_path = os.path.join(root, p)
        files = [
            f
            for f in os.listdir(k_path)
            if ".py" in f and "_base" not in f and "conductivity" in f
        ]
        files.sort()
        funcs = [pybamm.load_function(os.path.join(k_path, f)) for f in files]
        T_ref = 298.15
        T = T_ref + 30.0
        c = 1000.0
        k = [np.around(f(c, T).value, 6) for f in funcs]
        self.assertEqual(k, [1.839786, 1.361015, 0.750259])
        T += 20
        k = [np.around(f(c, T).value, 6) for f in funcs]
        self.assertEqual(k, [2.292425, 1.664438, 0.880755])

        chemistry = pybamm.parameter_sets.Chen2020
        param = pybamm.ParameterValues(chemistry=chemistry)
        param["Electrolyte conductivity [S.m-1]"] = funcs[0]
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.set_parameters()
        sim.build()

    def test_electrolyte_diffusivity(self):
        root = pybamm.root_dir()
        p = "pybamm/input/parameters/lithium-ion/electrolytes/lipf6_Landesfeind2019"
        d_path = os.path.join(root, p)
        files = [
            f
            for f in os.listdir(d_path)
            if ".py" in f and "_base" not in f and "diffusivity" in f
        ]
        files.sort()
        funcs = [pybamm.load_function(os.path.join(d_path, f)) for f in files]
        T_ref = 298.15
        T = T_ref + 30.0
        c = 1000.0
        D = [np.around(f(c, T).value, 16) for f in funcs]
        self.assertEqual(D, [5.796505e-10, 5.417881e-10, 5.608856e-10])
        T += 20
        D = [np.around(f(c, T).value, 16) for f in funcs]
        self.assertEqual(D, [8.5992e-10, 7.752815e-10, 7.907549e-10])

        chemistry = pybamm.parameter_sets.Chen2020
        param = pybamm.ParameterValues(chemistry=chemistry)
        param["Electrolyte diffusivity [m2.s-1]"] = funcs[0]
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.set_parameters()
        sim.build()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
