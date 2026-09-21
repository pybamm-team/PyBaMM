#
# Tests for the lithium-ion half-cell DFN model
#

import numpy as np

import pybamm
from tests import BaseUnitTestLithiumIonHalfCell


class TestDFNHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.DFN

    def test_initial_lithium_inventory(self):
        options = {"working electrode": "positive"}
        model = self.model(options)
        parameter_values = pybamm.ParameterValues("Xu2019")
        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            [0, 1]
        )

        area = (
            parameter_values["Electrode width [m]"]
            * parameter_values["Electrode height [m]"]
            * parameter_values[
                "Number of electrodes connected in parallel to make a cell"
            ]
        )
        expected_electrolyte_lithium = (
            area
            * parameter_values["Initial concentration in electrolyte [mol.m-3]"]
            * (
                parameter_values["Separator porosity"]
                * parameter_values["Separator thickness [m]"]
                + parameter_values["Positive electrode porosity"]
                * parameter_values["Positive electrode thickness [m]"]
            )
        )

        electrolyte_lithium = solution["Total lithium in electrolyte [mol]"](0)
        particle_lithium = solution["Total lithium in particles [mol]"](0)
        np.testing.assert_allclose(
            electrolyte_lithium, expected_electrolyte_lithium, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            parameter_values.evaluate(model.param.n_Li_e_init),
            expected_electrolyte_lithium,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            solution["Total lithium [mol]"](0),
            particle_lithium + expected_electrolyte_lithium,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            solution["Total lithium lost [mol]"](0), 0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            solution["Total lithium lost from electrolyte [mol]"](0),
            0,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            solution["Loss of lithium inventory, including electrolyte [%]"](0),
            0,
            rtol=0,
            atol=1e-12,
        )
