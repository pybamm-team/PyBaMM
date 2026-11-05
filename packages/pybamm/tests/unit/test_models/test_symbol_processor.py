#
# Tests for the SymbolProcessor class
#
import pytest

import pybamm


class TestSymbolProcessor:
    def test_init(self):
        processor = pybamm.SymbolProcessor()
        assert processor.parameter_values is None
        assert processor.discretisation is None
        # Initially cannot process symbols (no param or disc set)
        assert processor.can_process_symbols is False
        assert bool(processor) is False

    def test_bool(self):
        processor = pybamm.SymbolProcessor()

        # False when nothing is set
        assert bool(processor) is False

        # Set up parameter values and discretisation
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Only parameter_values set - still False
        processor.parameter_values = param
        assert bool(processor) is False

        # Both set - True
        processor.discretisation = disc
        assert bool(processor) is True

    def test_set_parameter_values(self):
        processor = pybamm.SymbolProcessor()
        param = pybamm.ParameterValues("Chen2020")

        processor.parameter_values = param
        assert processor.parameter_values is not None
        # Should be a copy
        assert processor.parameter_values is not param

    def test_set_discretisation(self):
        processor = pybamm.SymbolProcessor()

        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        processor.discretisation = disc
        assert processor.discretisation is not None
        # Should be a copy
        assert processor.discretisation is not disc

    def test_invalid_parameter_values(self):
        processor = pybamm.SymbolProcessor()

        with pytest.raises(ValueError, match=r"`parameter_values` must be"):
            processor.parameter_values = "not a ParameterValues"

    def test_invalid_discretisation(self):
        processor = pybamm.SymbolProcessor()

        with pytest.raises(ValueError, match=r"`discretisation` must be"):
            processor.discretisation = "not a Discretisation"

    def test_disable(self):
        processor = pybamm.SymbolProcessor()

        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        processor.parameter_values = param
        processor.discretisation = disc
        assert processor.can_process_symbols is True

        processor.disable()
        assert processor.can_process_symbols is False

    def test_setting_twice_disables_processing(self):
        processor = pybamm.SymbolProcessor()

        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # First set
        processor.parameter_values = param
        processor.discretisation = disc
        assert processor.can_process_symbols is True

        # Setting parameter_values again disables processing
        processor.parameter_values = param
        assert processor.can_process_symbols is False

    def test_copy(self):
        processor = pybamm.SymbolProcessor()

        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        processor.parameter_values = param
        processor.discretisation = disc

        processor_copy = processor.copy()
        assert processor_copy is not processor
        assert processor_copy.can_process_symbols == processor.can_process_symbols
        assert processor_copy.parameter_values is not None
        assert processor_copy.discretisation is not None

    def test_call_without_setup(self):
        processor = pybamm.SymbolProcessor()
        symbol = pybamm.Scalar(1)

        with pytest.raises(ValueError, match=r"Cannot process a symbol"):
            processor("test", symbol)
