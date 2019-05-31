import pybamm
import unittest


class TestParticle(unittest.TestCase):
    def test_not_implemented(self):
        c = pybamm.Variable("c", domain=["negative particle", "positive particle"])
        model = pybamm.models.submodels.particle.Standard(None)

        with self.assertRaises(NotImplementedError):
            model.set_differential_system(c, None)

        d = pybamm.Variable("d", domain=["not a domain"])
        with self.assertRaises(pybamm.ModelError):
            model.set_differential_system(d, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
