#
# Tests for BaseModel.to_json and BaseModel.from_json round-trip
#

import json

import numpy as np
import pytest

import pybamm


def _assert_solution_y_almost_equal(sol_orig, sol_loaded, key_output="Voltage [V]"):
    """Assert solution state vector has same shape and key output is almost equal."""
    np.testing.assert_array_equal(sol_orig.y.shape, sol_loaded.y.shape)
    v_orig = sol_orig[key_output].data
    v_loaded = sol_loaded[key_output].data
    np.testing.assert_allclose(v_orig, v_loaded, rtol=5e-3, atol=1e-4)


def _minimal_custom_model():
    """Build a minimal custom BaseModel for round-trip tests."""
    model = pybamm.BaseModel(name="test_to_json_model")
    a = pybamm.Variable("a", domain="electrode")
    b = pybamm.Variable("b", domain="electrode")
    model.rhs = {a: b}
    model.initial_conditions = {a: pybamm.Scalar(1)}
    model.algebraic = {}
    model.boundary_conditions = {a: {"left": (pybamm.Scalar(0), "Dirichlet")}}
    model.events = [pybamm.Event("terminal", pybamm.Scalar(1) - b, "TERMINATION")]
    model.variables = {"a": a, "b": b}
    return model


class TestBaseModelToJson:
    def test_to_json_returns_dict_custom_base_model(self):
        """to_json() returns a dict with expected keys; round-trip via from_json."""
        model = _minimal_custom_model()
        param_dict = model.to_json()
        assert isinstance(param_dict, dict)
        assert "schema_version" in param_dict
        assert "pybamm_version" in param_dict
        assert "model" in param_dict
        assert param_dict["model"]["name"] == "test_to_json_model"

        loaded = pybamm.BaseModel.from_json(param_dict)
        assert loaded.name == model.name
        assert getattr(loaded, "options", {}) == getattr(model, "options", {})
        assert isinstance(loaded.rhs, dict)
        assert isinstance(loaded.variables, dict)
        assert "a" in loaded.variables and "b" in loaded.variables

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_to_json_save_and_round_trip_custom_base_model(self, tmp_path, compress):
        """to_json(filename) writes file; from_json(path) loads equivalent model."""
        model = _minimal_custom_model()
        file_path = tmp_path / "custom.json"
        result = model.to_json(str(file_path), compress=compress)
        assert result is not None
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert "model" in data or "compressed" in data

        loaded = pybamm.BaseModel.from_json(str(file_path))
        assert loaded.name == model.name
        assert getattr(loaded, "options", {}) == getattr(model, "options", {})
        assert isinstance(loaded.rhs, dict)
        assert isinstance(loaded.variables, dict)

    def test_to_json_requires_json_extension(self):
        """to_json(filename) raises if filename does not end with .json."""
        model = _minimal_custom_model()
        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            model.to_json("model.txt")

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_spm_round_trip_no_options(self, compress):
        """SPM round-trip via to_json/from_json; equivalent model, can be solved."""
        model = pybamm.lithium_ion.SPM()
        d = model.to_json(compress=compress)
        loaded = pybamm.BaseModel.from_json(d)
        assert loaded.name == model.name
        assert dict(getattr(loaded, "options", {})) == dict(
            getattr(model, "options", {})
        )

        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_geometry(geometry)
        t_eval = np.linspace(0, 100, 25)

        # Solve original model
        param.process_model(model)
        mesh_orig = pybamm.Mesh(
            geometry, model.default_submesh_types, model.default_var_pts
        )
        disc_orig = pybamm.Discretisation(mesh_orig, model.default_spatial_methods)
        disc_orig.process_model(model)
        solution_original = model.default_solver.solve(model, t_eval)

        # Solve loaded model (same geometry and param)
        param.process_model(loaded)
        mesh_loaded = pybamm.Mesh(
            geometry, loaded.default_submesh_types, loaded.default_var_pts
        )
        disc_loaded = pybamm.Discretisation(mesh_loaded, loaded.default_spatial_methods)
        disc_loaded.process_model(loaded)
        solution_loaded = loaded.default_solver.solve(loaded, t_eval)

        _assert_solution_y_almost_equal(solution_original, solution_loaded)

    @pytest.mark.parametrize(
        "compress", [False, True], ids=["uncompressed", "compressed"]
    )
    def test_spm_with_options_round_trip(self, compress):
        """SPM with options round-trip; same name and options, model solves."""
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPM(options=options)
        d = model.to_json(compress=compress)
        loaded = pybamm.BaseModel.from_json(d)
        assert loaded.name == model.name
        assert dict(getattr(loaded, "options", {})) == dict(
            getattr(model, "options", {})
        )

        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_geometry(geometry)
        t_eval = np.linspace(0, 100, 25)

        # Solve original model
        param.process_model(model)
        mesh_orig = pybamm.Mesh(
            geometry, model.default_submesh_types, model.default_var_pts
        )
        disc_orig = pybamm.Discretisation(mesh_orig, model.default_spatial_methods)
        disc_orig.process_model(model)
        solution_original = model.default_solver.solve(model, t_eval)

        # Solve loaded model (same geometry and param)
        param.process_model(loaded)
        mesh_loaded = pybamm.Mesh(
            geometry, loaded.default_submesh_types, loaded.default_var_pts
        )
        disc_loaded = pybamm.Discretisation(mesh_loaded, loaded.default_spatial_methods)
        disc_loaded.process_model(loaded)
        solution_loaded = loaded.default_solver.solve(loaded, t_eval)

        _assert_solution_y_almost_equal(solution_original, solution_loaded)


class TestBaseModelToConfig:
    """Tests for BaseModel to_config and from_config (wrapped format)."""

    def test_to_config_returns_type_and_model(self):
        """to_config() returns dict with 'type': 'custom' and 'model'."""
        model = _minimal_custom_model()
        config = model.to_config()
        assert isinstance(config, dict)
        assert config.get("type") == "custom"
        assert "model" in config
        model_data = config["model"]
        assert "schema_version" in model_data
        assert "pybamm_version" in model_data
        assert "model" in model_data
        assert model_data["model"]["name"] == "test_to_json_model"

    def test_to_config_filename_writes_and_from_config_loads(self, tmp_path):
        """to_config(filename=path) writes file; from_config(path) loads model."""
        model = _minimal_custom_model()
        file_path = tmp_path / "config.json"
        result = model.to_config(filename=str(file_path))
        assert result.get("type") == "custom"
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data.get("type") == "custom" and "model" in data

        loaded = pybamm.BaseModel.from_config(str(file_path))
        assert loaded.name == model.name
        assert getattr(loaded, "options", {}) == getattr(model, "options", {})
        assert "a" in loaded.variables and "b" in loaded.variables

    def test_from_config_round_trip_from_to_config(self):
        """from_config(model.to_config()) round-trips."""
        model = _minimal_custom_model()
        config = model.to_config()
        loaded = pybamm.BaseModel.from_config(config)
        assert loaded.name == model.name
        assert getattr(loaded, "options", {}) == getattr(model, "options", {})
        assert isinstance(loaded.rhs, dict)
        assert "a" in loaded.variables and "b" in loaded.variables

    def test_from_config_accepts_raw_to_json_dict(self):
        """from_config(model.to_json()) still works (backward compatibility)."""
        model = _minimal_custom_model()
        raw = model.to_json()
        loaded = pybamm.BaseModel.from_config(raw)
        assert loaded.name == model.name
        assert "a" in loaded.variables and "b" in loaded.variables

    def test_to_config_requires_json_extension(self):
        """to_config(filename=...) raises if filename does not end with .json."""
        model = _minimal_custom_model()
        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            model.to_config(filename="config.txt")

    def test_model_with_default_bounds_variables_produces_valid_json(self):
        """Model with Variables (default bounds) serialises to valid JSON (no Infinity)."""
        model = _minimal_custom_model()
        d = model.to_json()
        json_str = json.dumps(d)
        assert "Infinity" not in json_str
        loaded = pybamm.BaseModel.from_json(json.loads(json_str))
        assert loaded.name == model.name
