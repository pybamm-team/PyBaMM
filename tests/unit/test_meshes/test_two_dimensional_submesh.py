import numpy as np
import pytest

import pybamm


@pytest.fixture()
def x():
    x = pybamm.SpatialVariable("x", domain=["my 2d domain"], coord_sys="cartesian")
    return x


@pytest.fixture()
def y():
    return pybamm.SpatialVariable("y", domain=["my 2d domain"], coord_sys="cartesian")


@pytest.fixture()
def geometry(x, y):
    geometry = {
        "my 2d domain": {
            x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)},
            y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)},
        }
    }
    return geometry


class TestSubMesh2D:
    def test_submesh_2d_read_lims(self, x, y):
        """
        Test the read_lims method of SubMesh2D parent class
        """
        # Create a SubMesh2D instance to test the method
        edges_lr = np.array([0.0, 0.5, 1.0])
        edges_tb = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        submesh = pybamm.SubMesh2D(edges_lr, edges_tb, "cartesian")

        # Test with SpatialVariable objects and tabs
        lims_with_tabs = {
            x: {"min": 0.0, "max": 2.0},
            y: {"min": -1.0, "max": 3.0},
            "tabs": {"negative": {"x_centre": 0.1, "width": 0.05}},
        }

        # Make a copy since read_lims modifies the input dict
        lims_copy = lims_with_tabs.copy()
        spatial_var_lr, spatial_lims_lr, spatial_var_tb, spatial_lims_tb, tabs = (
            submesh.read_lims(lims_copy)
        )

        # Test that variables are returned in lr/tb order (first variable is lr, second is tb)
        # The parent class assumes the first variable in the dict is lr and second is tb
        expected_vars = list(lims_with_tabs.keys())[
            :2
        ]  # Get first two keys (excluding tabs)
        assert spatial_var_lr == expected_vars[0]
        assert spatial_var_tb == expected_vars[1]

        # Test that limits are returned correctly
        assert spatial_lims_lr["min"] == 0.0
        assert spatial_lims_lr["max"] == 2.0
        assert spatial_lims_tb["min"] == -1.0
        assert spatial_lims_tb["max"] == 3.0

        # Test that tabs are extracted correctly
        assert tabs is not None
        assert "negative" in tabs
        assert tabs["negative"]["x_centre"] == 0.1
        assert tabs["negative"]["width"] == 0.05

        # Test that tabs are removed from original dict
        assert "tabs" not in lims_copy

    def test_submesh_2d_read_lims_string_variables(self):
        """
        Test read_lims method with string variable names
        """
        # Create a SubMesh2D instance to test the method
        edges_lr = np.array([0.0, 1.0])
        edges_tb = np.array([0.0, 1.0])
        submesh = pybamm.SubMesh2D(edges_lr, edges_tb, "cartesian")

        # Test with string variable names
        lims_strings = {"x": {"min": 0.0, "max": 1.0}, "y": {"min": 0.0, "max": 2.0}}

        lims_copy = lims_strings.copy()
        spatial_var_lr, _spatial_lims_lr, spatial_var_tb, _spatial_lims_tb, tabs = (
            submesh.read_lims(lims_copy)
        )

        # Test that string variables are converted to SpatialVariable objects
        assert isinstance(spatial_var_lr, pybamm.SpatialVariable)
        assert isinstance(spatial_var_tb, pybamm.SpatialVariable)

        # Test that variable names are correct (first key becomes lr, second becomes tb)
        assert spatial_var_lr.name == "x"
        assert spatial_var_tb.name == "y"

        # Test that tabs is None when not provided
        assert tabs is None

    def test_submesh_2d_read_lims_no_tabs(self, x, y):
        """
        Test read_lims method without tabs
        """
        # Create a SubMesh2D instance to test the method
        edges_lr = np.array([0.0, 1.0])
        edges_tb = np.array([0.0, 1.0])
        submesh = pybamm.SubMesh2D(edges_lr, edges_tb, "cartesian")

        # Test without tabs
        lims_no_tabs = {x: {"min": 0.0, "max": 1.0}, y: {"min": 0.0, "max": 1.0}}

        lims_copy = lims_no_tabs.copy()
        spatial_var_lr, spatial_lims_lr, spatial_var_tb, spatial_lims_tb, tabs = (
            submesh.read_lims(lims_copy)
        )

        # Test that tabs is None
        assert tabs is None

        # Test that variables and limits are correct
        assert spatial_var_lr == x
        assert spatial_var_tb == y
        assert spatial_lims_lr["min"] == 0.0
        assert spatial_lims_lr["max"] == 1.0
        assert spatial_lims_tb["min"] == 0.0
        assert spatial_lims_tb["max"] == 1.0

    def test_submesh_2d_read_lims_error_cases(self):
        """
        Test error cases for read_lims method
        """
        # Create a SubMesh2D instance to test the method
        edges_lr = np.array([0.0, 1.0])
        edges_tb = np.array([0.0, 1.0])
        submesh = pybamm.SubMesh2D(edges_lr, edges_tb, "cartesian")

        # Test with wrong number of variables (too few)
        lims_too_few = {"x": {"min": 0.0, "max": 1.0}}
        with pytest.raises(
            pybamm.GeometryError, match=r"lims should only contain two variables"
        ):
            submesh.read_lims(lims_too_few)

        # Test with wrong number of variables (too many)
        lims_too_many = {
            "x": {"min": 0.0, "max": 1.0},
            "y": {"min": 0.0, "max": 1.0},
            "z": {"min": 0.0, "max": 1.0},
        }
        with pytest.raises(
            pybamm.GeometryError, match=r"lims should only contain two variables"
        ):
            submesh.read_lims(lims_too_many)

    def test_submesh_2d_read_lims_mixed_variable_types(self, x):
        """
        Test read_lims method with mixed variable types (SpatialVariable and string)
        """
        # Create a SubMesh2D instance to test the method
        edges_lr = np.array([0.0, 1.0])
        edges_tb = np.array([0.0, 1.0])
        submesh = pybamm.SubMesh2D(edges_lr, edges_tb, "cartesian")

        # Test with mixed variable types
        lims_mixed = {
            x: {"min": 0.0, "max": 1.0},  # SpatialVariable (becomes lr)
            "y": {"min": 0.0, "max": 2.0},  # string (becomes tb)
        }

        lims_copy = lims_mixed.copy()
        spatial_var_lr, _spatial_lims_lr, spatial_var_tb, _spatial_lims_tb, _tabs = (
            submesh.read_lims(lims_copy)
        )

        # Test that both are now SpatialVariable objects
        assert isinstance(spatial_var_lr, pybamm.SpatialVariable)
        assert isinstance(spatial_var_tb, pybamm.SpatialVariable)

        # Test that the first variable (lr) is the original x variable
        assert spatial_var_lr == x

        # Test that the second variable (tb) is converted from string
        assert spatial_var_tb.name == "y"


class TestUniform2DSubMesh:
    def test_exceptions(self):
        lims = {"a": 1}
        with pytest.raises(pybamm.GeometryError):
            pybamm.Uniform2DSubMesh(lims, None)

    def test_symmetric_mesh_creation_no_parameters(self, x, y, geometry):
        submesh_types = {"my 2d domain": pybamm.Uniform2DSubMesh}
        var_pts = {x: 20, y: 20}

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        assert mesh["my 2d domain"].edges_lr[0] == 0
        assert mesh["my 2d domain"].edges_lr[-1] == 2
        assert mesh["my 2d domain"].edges_tb[0] == 0
        assert mesh["my 2d domain"].edges_tb[-1] == 1

        # check number of edges and nodes
        assert len(mesh["my 2d domain"].nodes_lr) == var_pts[x]
        assert len(mesh["my 2d domain"].nodes_tb) == var_pts[y]
        assert (
            len(mesh["my 2d domain"].edges_lr) == len(mesh["my 2d domain"].nodes_lr) + 1
        )
        assert (
            len(mesh["my 2d domain"].edges_tb) == len(mesh["my 2d domain"].nodes_tb) + 1
        )

    def test_uniform_2d_submesh_properties(self, x, y):
        """
        Test comprehensive properties of Uniform2DSubMesh
        """
        # Test with different limits and point counts
        lims = {x: {"min": 0.0, "max": 2.0}, y: {"min": -1.0, "max": 3.0}}
        npts = {x.name: 10, y.name: 15}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        # Test basic properties
        assert submesh.npts_lr == 10
        assert submesh.npts_tb == 15
        assert submesh.npts == 10 * 15  # Total points
        assert submesh.dimension == 2
        assert submesh.coord_sys == "cartesian"

        # Test edges
        assert len(submesh.edges_lr) == 11  # npts + 1
        assert len(submesh.edges_tb) == 16  # npts + 1
        assert submesh.edges_lr[0] == 0.0
        assert submesh.edges_lr[-1] == 2.0
        assert submesh.edges_tb[0] == -1.0
        assert submesh.edges_tb[-1] == 3.0

        # Test nodes (cell centers)
        assert len(submesh.nodes_lr) == 10
        assert len(submesh.nodes_tb) == 15

        # Test uniform spacing
        expected_dx = 2.0 / 10  # (max - min) / npts
        expected_dy = 4.0 / 15  # (max - min) / npts

        np.testing.assert_array_almost_equal(submesh.d_edges_lr, expected_dx)
        np.testing.assert_array_almost_equal(submesh.d_edges_tb, expected_dy)

        # Test that nodes are at cell centers
        expected_nodes_lr = np.linspace(
            0.0 + expected_dx / 2, 2.0 - expected_dx / 2, 10
        )
        expected_nodes_tb = np.linspace(
            -1.0 + expected_dy / 2, 3.0 - expected_dy / 2, 15
        )

        np.testing.assert_array_almost_equal(submesh.nodes_lr, expected_nodes_lr)
        np.testing.assert_array_almost_equal(submesh.nodes_tb, expected_nodes_tb)

        # Test d_nodes (spacing between node centers)
        expected_d_nodes_lr = np.full(9, expected_dx)  # npts - 1
        expected_d_nodes_tb = np.full(14, expected_dy)  # npts - 1

        np.testing.assert_array_almost_equal(submesh.d_nodes_lr, expected_d_nodes_lr)
        np.testing.assert_array_almost_equal(submesh.d_nodes_tb, expected_d_nodes_tb)

    def test_uniform_2d_submesh_with_tabs(self, x, y):
        """
        Test Uniform2DSubMesh with tab information
        """
        lims = {
            x: {"min": 0.0, "max": 1.0},
            y: {"min": 0.0, "max": 1.0},
            "tabs": {"negative": {"x_centre": 0.1, "width": 0.05}},
        }
        npts = {x.name: 20, y.name: 20}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        # Test that tabs are preserved
        assert submesh.tabs is not None
        assert "negative" in submesh.tabs
        assert submesh.tabs["negative"]["x_centre"] == 0.1
        assert submesh.tabs["negative"]["width"] == 0.05

    def test_uniform_2d_submesh_coordinate_system_mismatch(self, x):
        """
        Test that coordinate system mismatch raises an error
        """
        # Create a y variable with different coordinate system
        y_cylindrical = pybamm.SpatialVariable(
            "y", domain=["my 2d domain"], coord_sys="cylindrical polar"
        )

        lims = {x: {"min": 0.0, "max": 1.0}, y_cylindrical: {"min": 0.0, "max": 1.0}}
        npts = {x.name: 10, y_cylindrical.name: 10}

        with pytest.raises(
            pybamm.GeometryError, match=r"Coordinate systems must be the same"
        ):
            pybamm.Uniform2DSubMesh(lims, npts)

    def test_uniform_2d_submesh_string_variables(self):
        """
        Test Uniform2DSubMesh with string variable names
        """
        lims = {"x": {"min": 0.0, "max": 1.0}, "y": {"min": 0.0, "max": 1.0}}
        npts = {"x": 5, "y": 8}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        assert submesh.npts_lr == 5
        assert submesh.npts_tb == 8
        assert submesh.npts == 40

    def test_uniform_2d_submesh_edge_cases(self, x, y):
        """
        Test edge cases for Uniform2DSubMesh
        """
        # Test with minimum number of points
        lims = {x: {"min": 0.0, "max": 1.0}, y: {"min": 0.0, "max": 1.0}}
        npts = {x.name: 1, y.name: 1}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        assert submesh.npts_lr == 1
        assert submesh.npts_tb == 1
        assert len(submesh.edges_lr) == 2
        assert len(submesh.edges_tb) == 2
        assert len(submesh.nodes_lr) == 1
        assert len(submesh.nodes_tb) == 1

        # Test that single node is at center
        assert submesh.nodes_lr[0] == 0.5
        assert submesh.nodes_tb[0] == 0.5

    def test_uniform_2d_submesh_ghost_cells(self, x, y):
        """
        Test ghost cell creation for Uniform2DSubMesh
        """
        lims = {x: {"min": 0.0, "max": 1.0}, y: {"min": 0.0, "max": 1.0}}
        npts = {x.name: 5, y.name: 5}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        # Test left ghost cell
        left_ghost = submesh.create_ghost_cell("left")
        assert len(left_ghost.edges_lr) == 2
        assert len(left_ghost.edges_tb) == len(submesh.edges_tb)
        assert (
            left_ghost.edges_lr[1] == submesh.edges_lr[0]
        )  # Ghost cell connects to original

        # Test right ghost cell
        right_ghost = submesh.create_ghost_cell("right")
        assert len(right_ghost.edges_lr) == 2
        assert len(right_ghost.edges_tb) == len(submesh.edges_tb)
        assert (
            right_ghost.edges_lr[0] == submesh.edges_lr[-1]
        )  # Ghost cell connects to original

        # Test top ghost cell
        top_ghost = submesh.create_ghost_cell("top")
        assert len(top_ghost.edges_lr) == len(submesh.edges_lr)
        assert len(top_ghost.edges_tb) == 2
        assert (
            top_ghost.edges_tb[1] == submesh.edges_tb[0]
        )  # Ghost cell connects to original

        # Test bottom ghost cell
        bottom_ghost = submesh.create_ghost_cell("bottom")
        assert len(bottom_ghost.edges_lr) == len(submesh.edges_lr)
        assert len(bottom_ghost.edges_tb) == 2
        assert (
            bottom_ghost.edges_tb[0] == submesh.edges_tb[-1]
        )  # Ghost cell connects to original

        # Test invalid side
        with pytest.raises(ValueError, match=r"Invalid side"):
            submesh.create_ghost_cell("invalid")

    def test_uniform_2d_submesh_json_serialization(self, x, y):
        """
        Test JSON serialization of Uniform2DSubMesh
        """
        lims = {
            x: {"min": 0.0, "max": 1.0},
            y: {"min": 0.0, "max": 1.0},
            "tabs": {"negative": {"x_centre": 0.1}},
        }
        npts = {x.name: 3, y.name: 4}

        submesh = pybamm.Uniform2DSubMesh(lims, npts)

        # Test JSON conversion - now that the bug is fixed
        json_dict = submesh.to_json()

        assert "edges_lr" in json_dict
        assert "edges_tb" in json_dict
        assert "coord_sys" in json_dict
        assert "tabs" in json_dict

        # Test that edges are correctly serialized
        np.testing.assert_array_almost_equal(json_dict["edges_lr"], submesh.edges_lr)
        np.testing.assert_array_almost_equal(json_dict["edges_tb"], submesh.edges_tb)

        # Test that coordinate system is correct
        assert json_dict["coord_sys"] == "cartesian"

        # Test that tabs are preserved
        assert json_dict["tabs"]["negative"]["x_centre"] == 0.1
