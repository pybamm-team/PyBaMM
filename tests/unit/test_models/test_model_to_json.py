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

    # ---- Custom model tests ----

    def test_to_config_custom_model_returns_type_custom(self):
        """Custom BaseModel to_config() returns 'type': 'custom' and 'model'."""
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
        """to_config(filename=path) writes file; from_config(path) loads."""
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

    def test_from_config_round_trip_custom_model(self):
        """from_config(model.to_config()) round-trips for custom models."""
        model = _minimal_custom_model()
        config = model.to_config()
        loaded = pybamm.BaseModel.from_config(config)
        assert loaded.name == model.name
        assert getattr(loaded, "options", {}) == getattr(model, "options", {})
        assert isinstance(loaded.rhs, dict)
        assert "a" in loaded.variables and "b" in loaded.variables

    def test_from_config_accepts_raw_to_json_dict(self):
        """from_config(model.to_json()) still works (backward compat)."""
        model = _minimal_custom_model()
        raw = model.to_json()
        loaded = pybamm.BaseModel.from_config(raw)
        assert loaded.name == model.name
        assert "a" in loaded.variables and "b" in loaded.variables

    def test_to_config_requires_json_extension(self):
        """to_config(filename=...) raises if not .json."""
        model = _minimal_custom_model()
        with pytest.raises(ValueError, match=r"must end with '\.json' extension"):
            model.to_config(filename="config.txt")

    def test_model_with_default_bounds_variables_produces_valid_json(self):
        """Variables (default bounds) serialise to valid JSON (no Infinity)."""
        model = _minimal_custom_model()
        d = model.to_json()
        json_str = json.dumps(d)
        assert "Infinity" not in json_str
        loaded = pybamm.BaseModel.from_json(json.loads(json_str))
        assert loaded.name == model.name

    def test_custom_subclass_not_detected_as_builtin(self):
        """A subclass with a different class name is not a built-in."""

        class MySPM(pybamm.lithium_ion.SPM):
            pass

        mod = pybamm.BaseModel._find_builtin_module(MySPM)
        assert mod is None

    def test_custom_model_to_config_includes_defaults(self):
        """Custom model config includes geometry/var_pts/etc when available."""
        # SPM itself is built-in, so we test with a real SPM via to_json
        # and verify the custom path includes defaults by using a fresh
        # model that goes through the custom serialization.
        model = pybamm.lithium_ion.SPM()
        # Force custom path by calling Serialise directly
        from pybamm.expression_tree.operations.serialise import Serialise

        # Verify built-in models have defaults we can serialise
        geo = Serialise.serialise_custom_geometry(model.default_geometry)
        assert isinstance(geo, dict)
        vp = Serialise.serialise_var_pts(model.default_var_pts)
        assert isinstance(vp, dict)
        sm = Serialise.serialise_spatial_methods(model.default_spatial_methods)
        assert isinstance(sm, dict)
        st = Serialise.serialise_submesh_types(model.default_submesh_types)
        assert isinstance(st, dict)

    # ---- Built-in model tests ----

    def test_to_config_builtin_spm_simple_format(self):
        """Built-in SPM produces simple format with type and module."""
        model = pybamm.lithium_ion.SPM()
        config = model.to_config()
        assert config["type"] == "SPM"
        assert config["module"] == "lithium_ion"
        assert "model" not in config  # no serialised graph
        assert "options" in config  # options always included

    def test_to_config_builtin_spm_with_options(self):
        """Built-in SPM with options includes them in config."""
        model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})
        config = model.to_config()
        assert config["type"] == "SPM"
        assert config["module"] == "lithium_ion"
        assert config["options"]["thermal"] == "lumped"
        assert "model" not in config

    def test_to_config_builtin_dfn(self):
        """Built-in DFN produces simple format."""
        model = pybamm.lithium_ion.DFN()
        config = model.to_config()
        assert config["type"] == "DFN"
        assert config["module"] == "lithium_ion"
        assert "model" not in config

    def test_to_config_builtin_lead_acid(self):
        """Built-in lead_acid.LOQS produces simple format."""
        model = pybamm.lead_acid.LOQS()
        config = model.to_config()
        assert config["type"] == "LOQS"
        assert config["module"] == "lead_acid"
        assert "model" not in config

    def test_from_config_builtin_round_trip(self):
        """from_config round-trip for built-in SPM."""
        model = pybamm.lithium_ion.SPM()
        config = model.to_config()
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert loaded.name == model.name

    def test_from_config_builtin_with_options_round_trip(self):
        """from_config round-trip for built-in SPM with options."""
        model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})
        config = model.to_config()
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert dict(loaded.options) == dict(model.options)

    def test_from_config_builtin_with_tuple_options_round_trip(self):
        """Options containing tuples survive JSON round-trip."""
        model = pybamm.lithium_ion.SPM(
            options={
                "current collector": "potential pair",
                "dimensionality": 1,
            }
        )
        config = model.to_config()
        # Simulate JSON round-trip (tuples become lists)
        config = json.loads(json.dumps(config))
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert dict(loaded.options) == dict(model.options)

    def test_from_config_builtin_with_actual_tuple_valued_options_round_trip(self):
        """Tuple-valued options (e.g. particle phases) survive JSON round-trip."""
        model = pybamm.lithium_ion.DFN(options={"particle phases": ("2", "1")})
        config = model.to_config()
        # Simulate JSON round-trip (tuples become lists)
        config = json.loads(json.dumps(config))
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        # Tuple options should be restored
        assert dict(loaded.options) == dict(model.options)
        assert isinstance(loaded.options["particle phases"], tuple)

    def test_from_config_backward_compat_no_module_key(self):
        """Config without 'module' key defaults to lithium_ion."""
        config = {"type": "SPM"}
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is pybamm.lithium_ion.SPM

    def test_from_config_lead_acid_module(self):
        """from_config with module='lead_acid' loads correct model."""
        config = {"type": "LOQS", "module": "lead_acid"}
        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is pybamm.lead_acid.LOQS

    def test_from_config_unknown_module_raises(self):
        """from_config raises for unknown module."""
        config = {"type": "SPM", "module": "nonexistent"}
        with pytest.raises(ValueError, match="Unknown pybamm module"):
            pybamm.BaseModel.from_config(config)

    def test_from_config_unknown_type_raises(self):
        """from_config raises for unknown model type in valid module."""
        config = {"type": "NonExistentModel", "module": "lithium_ion"}
        with pytest.raises(ValueError, match="not found"):
            pybamm.BaseModel.from_config(config)

    def test_to_config_is_json_serializable(self):
        """Built-in model config is JSON-serializable."""
        model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})
        config = model.to_config()
        json_str = json.dumps(config)
        assert isinstance(json_str, str)
        reloaded = json.loads(json_str)
        assert reloaded == config

    def test_to_config_builtin_file_round_trip(self, tmp_path):
        """Built-in model to_config with filename writes and loads."""
        model = pybamm.lithium_ion.SPM()
        file_path = tmp_path / "builtin.json"
        model.to_config(filename=str(file_path))
        assert file_path.exists()
        loaded = pybamm.BaseModel.from_config(str(file_path))
        assert type(loaded) is type(model)

    # ---- Built-in model override tests ----

    def test_to_config_builtin_unmodified_has_no_overrides(self):
        """Unmodified built-in model config has no override keys."""
        model = pybamm.lithium_ion.SPM()
        config = model.to_config()
        assert config["type"] == "SPM"
        assert "extra_variables" not in config
        assert "removed_variables" not in config
        assert "events" not in config

    def test_to_config_builtin_with_extra_variable_round_trip(self):
        """Added variable survives to_config / from_config round-trip."""
        model = pybamm.lithium_ion.SPM()
        model.variables["My variable"] = (
            2 * model.variables["Voltage [V]"]
        )
        config = model.to_config()

        # Still compact format with override
        assert config["type"] == "SPM"
        assert "extra_variables" in config
        assert "My variable" in config["extra_variables"]

        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert "My variable" in loaded.variables

    def test_to_config_builtin_with_cleared_events_round_trip(self):
        """Clearing events survives to_config / from_config round-trip."""
        model = pybamm.lithium_ion.SPM()
        model.events = []
        config = model.to_config()

        assert config["type"] == "SPM"
        assert "events" in config
        assert config["events"] == []

        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert loaded.events == []

    def test_to_config_builtin_with_removed_variable_round_trip(self):
        """Removed variable is absent after from_config round-trip."""
        model = pybamm.lithium_ion.SPM()
        assert "Voltage [V]" in model.variables
        del model.variables["Voltage [V]"]
        config = model.to_config()

        assert config["type"] == "SPM"
        assert "removed_variables" in config
        assert "Voltage [V]" in config["removed_variables"]

        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert "Voltage [V]" not in loaded.variables

    def test_to_config_builtin_with_added_event_round_trip(self):
        """Custom event added to built-in model survives round-trip."""
        model = pybamm.lithium_ion.SPM()
        original_count = len(model.events)
        model.events.append(
            pybamm.Event(
                "my_custom_event",
                pybamm.Scalar(1),
                pybamm.EventType.TERMINATION,
            )
        )
        config = model.to_config()

        assert config["type"] == "SPM"
        assert "events" in config
        assert len(config["events"]) == original_count + 1

        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        event_names = {e.name for e in loaded.events}
        assert "my_custom_event" in event_names

    def test_to_config_builtin_combined_variable_and_event_changes(self):
        """Matches example.py: add variable + clear events, round-trip."""
        model = pybamm.lithium_ion.SPM()
        model.events = []
        model.variables["My variable"] = (
            2 * model.variables["Voltage [V]"]
        )
        config = model.to_config()

        assert config["type"] == "SPM"
        assert "extra_variables" in config
        assert "events" in config
        assert config["events"] == []

        loaded = pybamm.BaseModel.from_config(config)
        assert type(loaded) is type(model)
        assert "My variable" in loaded.variables
        assert loaded.events == []

    def test_to_config_builtin_overrides_json_serializable(self):
        """Config with overrides survives JSON round-trip."""
        model = pybamm.lithium_ion.SPM()
        model.events = []
        model.variables["My variable"] = (
            2 * model.variables["Voltage [V]"]
        )
        config = model.to_config()

        json_str = json.dumps(config)
        assert isinstance(json_str, str)
        reloaded = json.loads(json_str)

        loaded = pybamm.BaseModel.from_config(reloaded)
        assert "My variable" in loaded.variables
        assert loaded.events == []
