#
# Tests for the Thevenin equivalant circuit model
#

import pybamm
import pytest


class TestThevenin:
    def test_standard_model(self):
        model = pybamm.equivalent_circuit.Thevenin()
        model.check_well_posedness()

    def test_default_properties(self):
        model = pybamm.equivalent_circuit.Thevenin()
        x = model.variables["x ECMD"]

        # test var_pts
        assert model.default_var_pts == {x: 20}

        # test geometry
        assert model.default_geometry == {"ECMD particle": {x: {"min": 0, "max": 1}}}

        # test spatial methods
        assert isinstance(
            model.default_spatial_methods["ECMD particle"], pybamm.FiniteVolume
        )

        # test submesh types
        assert model.default_submesh_types == {"ECMD particle": pybamm.Uniform1DSubMesh}

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

        with pytest.raises(pybamm.OptionError, match="natural numbers"):
            options = {"number of rc elements": -1}
            model = pybamm.equivalent_circuit.Thevenin(options=options)
            model.check_well_posedness()

    def test_diffusion_element(self):
        options = {"diffusion element": "true"}
        model = pybamm.equivalent_circuit.Thevenin(options=options)
        model.check_well_posedness(post_discretisation=True)

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
            V = variables["Voltage [V]"]
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

    def test_raise_option_error(self):
        options = {"not an option": "something"}
        with pytest.raises(
            pybamm.OptionError, match="Option 'not an option' not recognised"
        ):
            pybamm.equivalent_circuit.Thevenin(options=options)

    def test_not_a_valid_option(self):
        options = {"operating mode": "not a valid option"}
        with pytest.raises(
            pybamm.OptionError, match="Option 'operating mode' must be one of"
        ):
            pybamm.equivalent_circuit.Thevenin(options=options)

    def test_get_default_parameters(self):
        model = pybamm.equivalent_circuit.Thevenin()
        values = model.default_parameter_values
        assert "Initial SoC" in list(values.keys())
        values.process_model(model)

    def test_get_default_quick_plot_variables(self):
        model = pybamm.equivalent_circuit.Thevenin()
        variables = model.default_quick_plot_variables
        assert "Current [A]" in variables
