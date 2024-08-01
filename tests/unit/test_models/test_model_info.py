#
# Tests getting model info
#
import pybamm


class TestModelInfo:
    def test_find_parameter_info(self):
        model = pybamm.lithium_ion.SPM()
        model.info("Negative particle diffusivity [m2.s-1]")
        model = pybamm.lithium_ion.SPMe()
        model.info("Negative particle diffusivity [m2.s-1]")
        model = pybamm.lithium_ion.DFN()
        model.info("Negative particle diffusivity [m2.s-1]")

        model.info("Not a parameter")
