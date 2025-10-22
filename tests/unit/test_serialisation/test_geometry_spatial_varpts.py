#
# Tests for serialization of geometry, spatial methods, and var_pts
#
import os
import tempfile
from pathlib import Path

import pytest

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise


class TestGeometrySerialization:
    def test_serialise_and_load_geometry(self):
        """Test saving and loading geometry to/from file."""
        # Create a custom geometry
        geometry = pybamm.battery_geometry()

        # Use temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_geometry.json"

            # Save geometry
            Serialise.save_custom_geometry(geometry, filename=str(filepath))
            assert filepath.exists()

            # Load geometry
            loaded_geometry = Serialise.load_custom_geometry(str(filepath))

            # Verify domains match
            assert set(loaded_geometry.keys()) == set(geometry.keys())

            # Verify spatial variables and their bounds
            for domain in geometry.keys():
                assert domain in loaded_geometry
                # Compare variable names
                orig_vars = {
                    (var.name if hasattr(var, "name") else var)
                    for var in geometry[domain].keys()
                    if var != "tabs"
                }
                loaded_vars = {
                    (var.name if hasattr(var, "name") else var)
                    for var in loaded_geometry[domain].keys()
                    if var != "tabs"
                }
                assert orig_vars == loaded_vars

    def test_serialise_and_load_geometry_dict(self):
        """Test serializing to dict and loading from dict."""
        # Create a custom geometry
        geometry = pybamm.battery_geometry()

        # Serialize to dict
        geometry_dict = Serialise.serialise_custom_geometry(geometry)

        # Verify structure
        assert "schema_version" in geometry_dict
        assert "pybamm_version" in geometry_dict
        assert "geometry" in geometry_dict

        # Load from dict
        loaded_geometry = Serialise.load_custom_geometry(geometry_dict)

        # Verify domains match
        assert set(loaded_geometry.keys()) == set(geometry.keys())

    def test_geometry_with_default_filename(self):
        """Test geometry saving with auto-generated filename."""
        geometry = pybamm.battery_geometry()

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Save with no filename (auto-generate)
            Serialise.save_custom_geometry(geometry)

            # Check a file was created
            json_files = list(Path(tmpdir).glob("geometry_*.json"))
            assert len(json_files) == 1

    def test_geometry_invalid_extension(self):
        """Test that non-.json extension raises error."""
        geometry = pybamm.battery_geometry()

        with pytest.raises(ValueError, match="must end with '.json' extension"):
            Serialise.save_custom_geometry(geometry, filename="test.txt")


class TestSpatialMethodsSerialization:
    def test_serialise_and_load_spatial_methods(self):
        """Test saving and loading spatial methods to/from file."""
        # Create spatial methods dict
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_spatial_methods.json"

            # Save spatial methods
            Serialise.save_spatial_methods(spatial_methods, filename=str(filepath))
            assert filepath.exists()

            # Load spatial methods
            loaded_methods = Serialise.load_spatial_methods(str(filepath))

            # Verify domains match
            assert set(loaded_methods.keys()) == set(spatial_methods.keys())

            # Verify class types match
            for domain in spatial_methods.keys():
                assert isinstance(loaded_methods[domain], type(spatial_methods[domain]))

            # Verify options are preserved
            for domain in spatial_methods.keys():
                if hasattr(spatial_methods[domain], "options"):
                    assert (
                        loaded_methods[domain].options
                        == spatial_methods[domain].options
                    )

    def test_serialise_and_load_spatial_methods_dict(self):
        """Test serializing to dict and loading from dict."""
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }

        # Serialize to dict
        methods_dict = Serialise.serialise_spatial_methods(spatial_methods)

        # Verify structure
        assert "schema_version" in methods_dict
        assert "pybamm_version" in methods_dict
        assert "spatial_methods" in methods_dict

        # Load from dict
        loaded_methods = Serialise.load_spatial_methods(methods_dict)

        # Verify domains match
        assert set(loaded_methods.keys()) == set(spatial_methods.keys())

    def test_spatial_methods_with_options(self):
        """Test that custom options are preserved."""
        # Create spatial method with custom options
        custom_options = {
            "extrapolation": {
                "order": {"gradient": "linear", "value": "quadratic"},
                "use bcs": True,
            }
        }
        spatial_methods = {"macroscale": pybamm.FiniteVolume(options=custom_options)}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_methods.json"

            # Save and load
            Serialise.save_spatial_methods(spatial_methods, filename=str(filepath))
            loaded_methods = Serialise.load_spatial_methods(str(filepath))

            # Verify options are preserved
            assert loaded_methods["macroscale"].options == custom_options

    def test_spatial_methods_invalid_class(self):
        """Test error handling for invalid spatial method class."""
        # Create invalid spatial methods data
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
            "spatial_methods": {
                "macroscale": {
                    "class": "NonExistentMethod",
                    "module": "pybamm.spatial_methods.finite_volume",
                    "options": {},
                }
            },
        }

        with pytest.raises(ImportError):
            Serialise.load_spatial_methods(invalid_data)


class TestVarPtsSerialization:
    def test_serialise_and_load_var_pts(self):
        """Test saving and loading var_pts to/from file."""
        # Create var_pts with string keys
        var_pts = {
            "x_n": 20,
            "x_s": 25,
            "x_p": 30,
            "r_n": 15,
            "r_p": 15,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_var_pts.json"

            # Save var_pts
            Serialise.save_var_pts(var_pts, filename=str(filepath))
            assert filepath.exists()

            # Load var_pts
            loaded_var_pts = Serialise.load_var_pts(str(filepath))

            # Verify all keys and values match
            assert loaded_var_pts == var_pts

    def test_serialise_and_load_var_pts_dict(self):
        """Test serializing to dict and loading from dict."""
        var_pts = {"x_n": 20, "x_s": 25, "x_p": 30}

        # Serialize to dict
        var_pts_dict = Serialise.serialise_var_pts(var_pts)

        # Verify structure
        assert "schema_version" in var_pts_dict
        assert "pybamm_version" in var_pts_dict
        assert "var_pts" in var_pts_dict

        # Load from dict
        loaded_var_pts = Serialise.load_var_pts(var_pts_dict)

        # Verify match
        assert loaded_var_pts == var_pts

    def test_var_pts_with_spatial_variables(self):
        """Test var_pts with SpatialVariable keys."""
        # Create var_pts with SpatialVariable keys
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        x_s = pybamm.SpatialVariable("x_s", domain="separator")

        var_pts = {
            x_n: 20,
            x_s: 25,
            "r_p": 15,  # Mix with string key
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_var_pts.json"

            # Save var_pts
            Serialise.save_var_pts(var_pts, filename=str(filepath))

            # Load var_pts (will have all string keys)
            loaded_var_pts = Serialise.load_var_pts(str(filepath))

            # Verify all keys are converted to strings
            expected = {"x_n": 20, "x_s": 25, "r_p": 15}
            assert loaded_var_pts == expected

    def test_var_pts_mixed_keys(self):
        """Test var_pts with both string and SpatialVariable keys."""
        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        var_pts = {
            x_n: 20,
            "x_s": 25,
            "x_p": 30,
        }

        # Serialize to dict
        var_pts_dict = Serialise.serialise_var_pts(var_pts)

        # All keys should be strings
        assert set(var_pts_dict["var_pts"].keys()) == {"x_n", "x_s", "x_p"}

    def test_var_pts_with_default_filename(self):
        """Test var_pts saving with auto-generated filename."""
        var_pts = {"x_n": 20}

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Save with no filename (auto-generate)
            Serialise.save_var_pts(var_pts)

            # Check a file was created
            json_files = list(Path(tmpdir).glob("var_pts_*.json"))
            assert len(json_files) == 1


class TestSubmeshTypesSerialization:
    def test_serialise_and_load_submesh_types(self):
        """Test saving and loading submesh types to/from file."""
        # Create submesh types dict
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_submesh_types.json"

            # Save submesh types
            Serialise.save_submesh_types(submesh_types, filename=str(filepath))
            assert filepath.exists()

            # Load submesh types
            loaded_submesh_types = Serialise.load_submesh_types(str(filepath))

            # Verify domains match
            assert set(loaded_submesh_types.keys()) == set(submesh_types.keys())

            # Verify class types match
            for domain in submesh_types.keys():
                assert isinstance(
                    loaded_submesh_types[domain], type(submesh_types[domain])
                )

    def test_serialise_and_load_submesh_types_dict(self):
        """Test serializing to dict and loading from dict."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        # Serialize to dict
        submesh_dict = Serialise.serialise_submesh_types(submesh_types)

        # Verify structure
        assert "schema_version" in submesh_dict
        assert "pybamm_version" in submesh_dict
        assert "submesh_types" in submesh_dict

        # Load from dict
        loaded_submesh_types = Serialise.load_submesh_types(submesh_dict)

        # Verify domains match
        assert set(loaded_submesh_types.keys()) == set(submesh_types.keys())

    def test_submesh_types_with_default_filename(self):
        """Test submesh types saving with auto-generated filename."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Save with no filename (auto-generate)
            Serialise.save_submesh_types(submesh_types)

            # Check a file was created
            json_files = list(Path(tmpdir).glob("submesh_types_*.json"))
            assert len(json_files) == 1

    def test_submesh_types_invalid_class(self):
        """Test error handling for invalid submesh type class."""
        # Create invalid submesh types data
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
            "submesh_types": {
                "negative electrode": {
                    "class": "NonExistentMesh",
                    "module": "pybamm.meshes.zero_dimensional_submesh",
                }
            },
        }

        with pytest.raises(ImportError):
            Serialise.load_submesh_types(invalid_data)

    def test_submesh_types_invalid_extension(self):
        """Test that non-.json extension raises error."""
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        with pytest.raises(ValueError, match="must end with '.json' extension"):
            Serialise.save_submesh_types(submesh_types, filename="test.txt")


class TestErrorHandling:
    def test_invalid_schema_version_geometry(self):
        """Test that invalid schema version raises error."""
        invalid_data = {
            "schema_version": "99.0",
            "pybamm_version": pybamm.__version__,
            "geometry": {},
        }

        with pytest.raises(ValueError, match="Unsupported schema version"):
            Serialise.load_custom_geometry(invalid_data)

    def test_missing_geometry_section(self):
        """Test error when geometry section is missing."""
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match="Missing 'geometry' section"):
            Serialise.load_custom_geometry(invalid_data)

    def test_missing_spatial_methods_section(self):
        """Test error when spatial_methods section is missing."""
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match="Missing 'spatial_methods' section"):
            Serialise.load_spatial_methods(invalid_data)

    def test_missing_var_pts_section(self):
        """Test error when var_pts section is missing."""
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match="Missing 'var_pts' section"):
            Serialise.load_var_pts(invalid_data)

    def test_file_not_found_geometry(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_custom_geometry("nonexistent_file.json")

    def test_file_not_found_spatial_methods(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_spatial_methods("nonexistent_file.json")

    def test_file_not_found_var_pts(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_var_pts("nonexistent_file.json")

    def test_missing_submesh_types_section(self):
        """Test error when submesh_types section is missing."""
        invalid_data = {
            "schema_version": "1.0",
            "pybamm_version": pybamm.__version__,
        }

        with pytest.raises(KeyError, match="Missing 'submesh_types' section"):
            Serialise.load_submesh_types(invalid_data)

    def test_file_not_found_submesh_types(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            Serialise.load_submesh_types("nonexistent_file.json")
