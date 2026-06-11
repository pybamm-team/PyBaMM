import os

import numpy as np
import pytest

import pybamm


class TestScikitFemGenerator3D:
    def test_pouch_mesh_creation(self):
        try:
            from pybamm.meshes.scikit_fem_submeshes_3d import ScikitFemGenerator3D
        except ImportError:
            pytest.skip("scikit-fem not available")

        geometry = {
            "domain": {
                pybamm.standard_spatial_vars.x: {"min": 0, "max": 1},
                pybamm.standard_spatial_vars.y: {"min": 0, "max": 2},
                pybamm.standard_spatial_vars.z: {"min": 0, "max": 3},
            }
        }

        mesh_gen = ScikitFemGenerator3D("pouch", h=0.3)

        mesh = pybamm.Mesh(
            geometry,
            {"domain": mesh_gen},
            {
                pybamm.standard_spatial_vars.x: None,
                pybamm.standard_spatial_vars.y: None,
                pybamm.standard_spatial_vars.z: None,
            },
        )

        submesh = mesh["domain"]
        assert hasattr(submesh, "_skfem_mesh"), "Mesh should have _skfem_mesh attribute"
        assert hasattr(submesh, "nodes"), "Mesh should have nodes attribute"
        assert hasattr(submesh, "elements"), "Mesh should have elements attribute"
        assert submesh.dimension == 3, "Mesh should be 3D"
        assert submesh.npts > 0, "Mesh should have nodes"
        assert submesh.nodes.shape[1] == 3, "Nodes should be 3D coordinates"
        assert submesh.coord_sys == "cartesian", "Pouch mesh should be cartesian"

    def test_cylinder_mesh_creation(self):
        try:
            from pybamm.meshes.scikit_fem_submeshes_3d import ScikitFemGenerator3D
        except ImportError:
            pytest.skip("scikit-fem not available")

        r = pybamm.SpatialVariable("r", ["domain"], coord_sys="cylindrical polar")
        z = pybamm.SpatialVariable("z", ["domain"], coord_sys="cylindrical polar")

        geometry = {
            "domain": {
                r: {"min": 0.1, "max": 1.0},
                z: {"min": 0, "max": 2.0},
            }
        }

        mesh_gen = ScikitFemGenerator3D("cylinder", h=0.5)

        var_pts = {r: None, z: None}

        mesh = pybamm.Mesh(geometry, {"domain": mesh_gen}, var_pts)

        submesh = mesh["domain"]
        assert hasattr(submesh, "_skfem_mesh")
        assert submesh.dimension == 3
        assert submesh.coord_sys == "cylindrical polar"

        nodes = submesh.nodes
        x_coords, y_coords, z_coords = nodes.T
        radii = np.sqrt(x_coords**2 + y_coords**2)

        assert np.max(radii) < 1.0 + 1e-7
        assert np.min(radii) > 0.1 - 1e-7
        assert np.min(z_coords) >= 0.0 - 1e-7
        assert np.max(z_coords) <= 2.0 + 1e-7

    def test_invalid_geometry_type(self):
        try:
            from pybamm.meshes.scikit_fem_submeshes_3d import ScikitFemGenerator3D
        except ImportError:
            pytest.skip("scikit-fem not available")

        with pytest.raises(pybamm.GeometryError, match=r"geom_type must be one of"):
            ScikitFemGenerator3D("invalid_type", h=0.3)


class TestScikitFemSubMesh3D:
    def test_submesh_properties(self):
        try:
            from pybamm.meshes.scikit_fem_submeshes_3d import (
                ScikitFemGenerator3D,
                ScikitFemSubMesh3D,
            )
        except ImportError:
            pytest.skip("scikit-fem not available")

        geometry = {
            "domain": {
                pybamm.standard_spatial_vars.x: {"min": 0, "max": 1},
                pybamm.standard_spatial_vars.y: {"min": 0, "max": 2},
                pybamm.standard_spatial_vars.z: {"min": 0, "max": 3},
            }
        }

        mesh_gen = ScikitFemGenerator3D("pouch", h=0.4)
        mesh = pybamm.Mesh(
            geometry,
            {"domain": mesh_gen},
            {
                pybamm.standard_spatial_vars.x: None,
                pybamm.standard_spatial_vars.y: None,
                pybamm.standard_spatial_vars.z: None,
            },
        )

        submesh = mesh["domain"]

        assert isinstance(submesh, ScikitFemSubMesh3D), (
            "Should be ScikitFemSubMesh3D instance"
        )

        assert hasattr(submesh, "_skfem_mesh"), "Should have _skfem_mesh"
        assert hasattr(submesh, "nodes"), "Should have nodes"
        assert hasattr(submesh, "elements"), "Should have elements"
        assert hasattr(submesh, "npts"), "Should have npts"
        assert hasattr(submesh, "dimension"), "Should have dimension"
        assert hasattr(submesh, "coord_sys"), "Should have coord_sys"

        assert submesh.dimension == 3, "Should be 3D"
        assert submesh.npts == len(submesh.nodes), "npts should equal number of nodes"
        assert submesh.nodes.shape[1] == 3, "Nodes should be 3D"
        assert len(submesh.elements) > 0, "Should have elements"

    def test_serialization(self):
        try:
            from pybamm.meshes.scikit_fem_submeshes_3d import (
                ScikitFemGenerator3D,
                ScikitFemSubMesh3D,
            )
        except ImportError:
            pytest.skip("scikit-fem not available")

        geometry = {
            "domain": {
                pybamm.standard_spatial_vars.x: {"min": 0, "max": 1},
                pybamm.standard_spatial_vars.y: {"min": 0, "max": 2},
                pybamm.standard_spatial_vars.z: {"min": 0, "max": 3},
            }
        }

        mesh_gen = ScikitFemGenerator3D("pouch", h=0.5)
        mesh = pybamm.Mesh(
            geometry,
            {"domain": mesh_gen},
            {
                pybamm.standard_spatial_vars.x: None,
                pybamm.standard_spatial_vars.y: None,
                pybamm.standard_spatial_vars.z: None,
            },
        )

        submesh = mesh["domain"]

        json_data = submesh.to_json()
        assert isinstance(json_data, dict), "to_json should return dict"
        assert "mesh_type" in json_data, "Should have mesh_type"
        assert json_data["mesh_type"] == "ScikitFemSubMesh3D", (
            "Should identify as ScikitFemSubMesh3D"
        )
        assert "coord_sys" in json_data, "Should have coord_sys"
        assert "nodes" in json_data, "Should have nodes"
        assert "elements" in json_data, "Should have elements"

        restored_submesh = ScikitFemSubMesh3D._from_json(json_data)
        assert isinstance(restored_submesh, ScikitFemSubMesh3D), (
            "Should restore to ScikitFemSubMesh3D"
        )
        assert restored_submesh.coord_sys == submesh.coord_sys, (
            "Should preserve coord_sys"
        )
        assert restored_submesh.npts == submesh.npts, "Should preserve npts"

    def test_invalid_geometry_type(self):
        with pytest.raises(pybamm.GeometryError, match=r"geom_type must be one of"):
            pybamm.ScikitFemGenerator3D("invalid_type", h=0.3)


MESH_DIR = os.path.join(os.path.dirname(__file__), "assets")
VALID_MESH_FILE = os.path.join(MESH_DIR, "test_mesh.msh")
NO_TETRA_MESH_FILE = os.path.join(MESH_DIR, "no_tetra_mesh.msh")
AMBIGUOUS_TAGS_MESH_FILE = os.path.join(MESH_DIR, "ambiguous_tags_mesh.msh")
NO_INTEGER_TAGS_MESH_FILE = os.path.join(MESH_DIR, "no_integer_tags_mesh.vtu")

requires_mesh_assets = pytest.mark.skipif(
    not os.path.exists(VALID_MESH_FILE),
    reason=f"Test mesh asset file not found: {VALID_MESH_FILE}",
)


class TestUserSuppliedMesh:
    def test_load_mesh_from_file_success(self):
        boundary_map = {"z_min": 1, "z_max": 2}
        domain_map = {"current collector": 5}

        skfem_mesh = pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
            VALID_MESH_FILE, boundary_map, domain_map
        )

        assert skfem_mesh.p.shape[1] > 0
        assert skfem_mesh.nelements > 0
        assert "current collector" in skfem_mesh.subdomains
        assert "z_min" in skfem_mesh.boundaries
        assert "z_max" in skfem_mesh.boundaries
        assert len(skfem_mesh.boundaries["z_min"]) > 0

    def test_load_file_not_found(self):
        with pytest.raises(pybamm.GeometryError, match=r"Could not read mesh file"):
            pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
                "non_existent_file.msh", {}, {}
            )

    def test_missing_tags(self, caplog):
        # Tag 99 does not exist in the file
        boundary_map = {"non_existent_boundary": 99}
        domain_map = {"non_existent_domain": 98}

        with caplog.at_level("WARNING", logger="pybamm"):
            skfem_mesh = pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
                VALID_MESH_FILE, boundary_map, domain_map
            )
            assert "No boundary facets found for 'non_existent_boundary'" in caplog.text
            assert "No elements found for domain 'non_existent_domain'" in caplog.text

        assert skfem_mesh.boundaries is None
        assert skfem_mesh.subdomains is None

    def test_user_supplied_generator(self):
        boundary_map = {"z_min": 1}
        domain_map = {"my_domain": 5}

        generator = pybamm.UserSuppliedSubmesh3D(
            file_path=VALID_MESH_FILE,
            boundary_mapping=boundary_map,
            domain_mapping=domain_map,
        )

        submesh = generator(lims=None, npts=None)

        assert isinstance(submesh, pybamm.ScikitFemSubMesh3D)
        assert submesh.npts > 0
        assert "z_min_dofs" in submesh.__dict__  # Check that dofs were created

    def test_integration_with_pybamm_mesh(self):
        domain_name = "current collector"
        boundary_map = {"z_min": 1, "z_max": 2}
        domain_map = {domain_name: 5}

        mesh_generator = pybamm.UserSuppliedSubmesh3D(
            file_path=VALID_MESH_FILE,
            boundary_mapping=boundary_map,
            domain_mapping=domain_map,
        )

        geometry = pybamm.Geometry({domain_name: {"x": {}, "y": {}, "z": {}}})
        submesh_types = {domain_name: mesh_generator}
        var_pts = {"x": None, "y": None, "z": None}

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        assert isinstance(mesh[domain_name], pybamm.ScikitFemSubMesh3D)
        assert mesh[domain_name].npts > 0
        assert "z_min" in mesh[domain_name]._skfem_mesh.boundaries

    def test_load_no_tetrahedra_error(self):
        with pytest.raises(
            pybamm.GeometryError, match=r"No tetrahedral elements found"
        ):
            pybamm.ScikitFemSubMesh3D.load_mesh_from_file(NO_TETRA_MESH_FILE, {}, {})

    def test_bad_user_tag_name_error(self):
        with pytest.raises(
            pybamm.GeometryError,
            match=r"User-specified domain tag name 'bad_tag' not found",
        ):
            pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
                VALID_MESH_FILE, {}, {}, domain_tag_name="bad_tag"
            )

    def test_no_integer_tag_error(self):
        with pytest.raises(
            pybamm.GeometryError,
            match=r"Could not automatically detect domain tag array",
        ):
            pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
                NO_INTEGER_TAGS_MESH_FILE, {}, {"domain": 1}
            )

    def test_internal_exception_handling(self, monkeypatch):
        def mock_error(*args, **kwargs):
            raise ValueError("Simulated internal error")

        monkeypatch.setattr(np, "where", mock_error)
        with pytest.raises(
            pybamm.GeometryError, match=r"Failed to extract tetrahedral elements"
        ):
            pybamm.ScikitFemSubMesh3D.load_mesh_from_file(
                VALID_MESH_FILE, {}, {"domain": 5}
            )
