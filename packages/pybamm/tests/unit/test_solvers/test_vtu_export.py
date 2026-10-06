import os
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import pybamm

vtk = pytest.importorskip("vtk")
from vtk.util.numpy_support import vtk_to_numpy


def _strip_mesh(x_min, x_max, n_cells, dimension):
    """A row of ``n_cells`` quads (2D) or hexahedra (3D) along x."""
    x = np.linspace(x_min, x_max, n_cells + 1)
    if dimension == 2:
        vertices = np.array([[xi, z] for z in (0.0, 1.0) for xi in x])
        n = n_cells + 1
        elements = [[i, i + 1, n + i + 1, n + i] for i in range(n_cells)]
    else:
        vertices = np.array(
            [[xi, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for xi in x]
        )
        n = n_cells + 1
        elements = [
            [
                *(i, i + 1, n + i + 1, n + i),
                *(2 * n + i, 2 * n + i + 1, 3 * n + i + 1, 3 * n + i),
            ]
            for i in range(n_cells)
        ]
    return pybamm.UnstructuredSubMesh(vertices, np.array(elements))


def _geometry(domains, dimension):
    axes = "xz" if dimension == 2 else "xyz"
    return {
        domain: {
            pybamm.SpatialVariable(axis, domain=domain): {
                "min": pybamm.Scalar(0),
                "max": pybamm.Scalar(1),
            }
            for axis in axes
        }
        for domain in domains
    }


def _two_region_solution(dimension):
    """Solution with variables over regions a and b, over a only, a vector and a 0D.

    Region a holds 2 cells on x in [0, 1] and region b 3 cells on x in [1, 2];
    ``y`` stores 1-5 (both regions), 6-7 (region a) at t = 0 and doubles by t = 2.
    """
    mesh_a = _strip_mesh(0.0, 1.0, 2, dimension)
    mesh_b = _strip_mesh(1.0, 2.0, 3, dimension)
    mesh_ab = pybamm.UnstructuredSubMesh.combine([mesh_a, mesh_b])

    model = pybamm.BaseModel()
    model._geometry = _geometry(["a", "b"], dimension)
    both = pybamm.StateVector(slice(0, 5), domain=["a", "b"]).with_mesh(mesh_ab)
    only_a = pybamm.StateVector(slice(5, 7), domain="a").with_mesh(mesh_a)
    components = [(k * both).with_mesh(mesh_ab) for k in range(1, dimension + 1)]
    vector = pybamm.VectorField(*components).with_mesh(mesh_ab)
    model.variables = {
        "Both regions [K]": both,
        "Region a only [V]": only_a,
        "Flux [A.m-2]": vector,
        "Voltage [V]": 4 - pybamm.t,
    }
    model.update_processed_variables(model.variables)

    t = np.array([0.0, 1.0, 2.0])
    y0 = np.arange(1.0, 8.0)
    y = np.asfortranarray(np.outer(y0, [1.0, 1.5, 2.0]))
    return pybamm.Solution(t, y, model, {}), mesh_ab


_VTK_CELL_TYPES = {
    vtk.VTK_TRIANGLE: "triangle",
    vtk.VTK_QUAD: "quad",
    vtk.VTK_TETRA: "tetra",
    vtk.VTK_HEXAHEDRON: "hexahedron",
}


def _read(directory, name):
    """Read a ``.vtu`` with VTK into a meshio-like namespace.

    meshio's reader cannot decode VTK's appended base64 data on Python 3.14.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.join(directory, name))
    reader.Update()
    grid = reader.GetOutput()

    cells_dict = {}
    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cells_dict.setdefault(_VTK_CELL_TYPES[cell.GetCellType()], []).append(ids)

    def arrays(data):
        return {
            data.GetArrayName(i): vtk_to_numpy(data.GetAbstractArray(i))
            for i in range(data.GetNumberOfArrays())
        }

    return types.SimpleNamespace(
        points=vtk_to_numpy(grid.GetPoints().GetData()),
        cells_dict={k: np.array(v) for k, v in cells_dict.items()},
        cell_data={k: [v] for k, v in arrays(grid.GetCellData()).items()},
        field_data=arrays(grid.GetFieldData()),
    )


class TestSaveVtu:
    @pytest.mark.parametrize("dimension", [2, 3])
    def test_round_trip_multi_region_and_partial_variables(self, tmp_path, dimension):
        solution, mesh_ab = _two_region_solution(dimension)
        variables = list(solution.all_models[0].variables)
        pvd = solution.save_vtu(tmp_path / "out.pvd", variables)

        assert pvd == os.fspath(tmp_path / "out.pvd")
        for i, t in enumerate(solution.t):
            vtu = _read(tmp_path, f"out/out_{i:04d}.vtu")
            factor = 1 + 0.5 * t

            cell_type = "quad" if dimension == 2 else "hexahedron"
            assert list(vtu.cells_dict) == [cell_type]
            assert len(vtu.cells_dict[cell_type]) == 5
            # 2D (x, z) meshes are written in the z = 0 plane
            np.testing.assert_allclose(
                vtu.points[:, :dimension].min(axis=0), np.zeros(dimension)
            )
            np.testing.assert_allclose(vtu.points[:, dimension:], 0.0)
            centroids = vtu.points[vtu.cells_dict[cell_type]].mean(axis=1)
            np.testing.assert_allclose(
                centroids[:, :dimension], mesh_ab.cell_centroids, atol=1e-12
            )

            both = vtu.cell_data["Both regions [K]"][0]
            np.testing.assert_allclose(both, factor * np.arange(1.0, 6.0))
            np.testing.assert_allclose(
                solution["Both regions [K]"](t=t).ravel(), both, rtol=1e-12
            )

            # cells outside the variable's region are blanked with NaN
            only_a = vtu.cell_data["Region a only [V]"][0]
            np.testing.assert_allclose(only_a[:2], factor * np.array([6.0, 7.0]))
            assert np.isnan(only_a[2:]).all()

            flux = vtu.cell_data["Flux [A.m-2]"][0]
            assert flux.shape == (5, 3)
            expected = np.zeros((5, 3))
            for k in range(dimension):
                expected[:, k] = (k + 1) * factor * np.arange(1.0, 6.0)
            np.testing.assert_allclose(flux, expected)

            # 0D variables and the time are stored as field data
            np.testing.assert_allclose(vtu.field_data["TimeValue"], [t])
            np.testing.assert_allclose(vtu.field_data["Voltage [V]"], [4 - t])

    def test_single_region_variables_share_cells(self, tmp_path):
        solution, _ = _two_region_solution(2)
        solution.save_vtu(tmp_path / "a", "Region a only [V]", t=0.0)

        vtu = _read(tmp_path, "a/a_0000.vtu")
        assert len(vtu.cells_dict["quad"]) == 2
        np.testing.assert_allclose(vtu.cell_data["Region a only [V]"][0], [6.0, 7.0])

    def test_disjoint_meshes_are_merged(self, tmp_path):
        nodes = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        mesh = pybamm.UnstructuredSubMesh(nodes, np.array([[0, 1, 2, 3]]))
        shifted_mesh = pybamm.UnstructuredSubMesh(
            nodes + np.array([2.0, 0.0, 0.0]), mesh.elements
        )
        model = pybamm.BaseModel()
        model._geometry = _geometry(["mesh", "shifted"], 3)
        model.variables = {
            "field": pybamm.StateVector(slice(0, 1), domain="mesh").with_mesh(mesh),
            "shifted": pybamm.StateVector(slice(1, 2), domain="shifted").with_mesh(
                shifted_mesh
            ),
        }
        model.update_processed_variables(model.variables)
        solution = pybamm.Solution(
            np.array([0.0, 1.0]),
            np.asfortranarray([[1.0, 2.0], [10.0, 20.0]]),
            model,
            {},
        )

        solution.save_vtu(tmp_path / "tet", ["field", "shifted"], t=[1.0])

        vtu = _read(tmp_path, "tet/tet_0000.vtu")
        assert len(vtu.cells_dict["tetra"]) == 2
        assert vtu.points.shape == (8, 3)
        np.testing.assert_allclose(vtu.cell_data["field"][0], [2.0, np.nan])
        np.testing.assert_allclose(vtu.cell_data["shifted"][0], [np.nan, 20.0])

    def test_pvd_indexes_vtu_files_by_time(self, tmp_path):
        solution, _ = _two_region_solution(2)
        pvd = solution.save_vtu(str(tmp_path / "series"), ["Both regions [K]"])

        assert pvd == str(tmp_path / "series.pvd")
        root = ET.parse(pvd).getroot()
        assert root.get("type") == "Collection"
        datasets = root.find("Collection").findall("DataSet")
        np.testing.assert_allclose(
            [float(d.get("timestep")) for d in datasets], solution.t
        )
        for i, dataset in enumerate(datasets):
            assert dataset.get("file") == f"series/series_{i:04d}.vtu"
            vtu = _read(tmp_path, dataset.get("file"))
            np.testing.assert_allclose(vtu.field_data["TimeValue"], [solution.t[i]])

    def test_output_times_are_interpolated(self, tmp_path):
        solution, _ = _two_region_solution(2)
        solution.save_vtu(
            tmp_path / "out", ["Both regions [K]", "Voltage [V]"], t=[0.5]
        )

        root = ET.parse(tmp_path / "out.pvd").getroot()
        datasets = root.find("Collection").findall("DataSet")
        assert [float(d.get("timestep")) for d in datasets] == [0.5]
        vtu = _read(tmp_path, "out/out_0000.vtu")
        np.testing.assert_allclose(
            vtu.cell_data["Both regions [K]"][0], 1.25 * np.arange(1.0, 6.0)
        )
        np.testing.assert_allclose(vtu.field_data["Voltage [V]"], [3.5])

    def test_invalid_inputs(self, tmp_path):
        solution, _ = _two_region_solution(2)
        path = tmp_path / "out"
        with pytest.raises(pybamm.OptionError, match=r"at least one variable"):
            solution.save_vtu(path, [])
        with pytest.raises(pybamm.OptionError, match=r"written alongside them"):
            solution.save_vtu(path, ["Voltage [V]"])
        with pytest.raises(pybamm.OptionError, match=r"time range"):
            solution.save_vtu(path, ["Both regions [K]"], t=[3.0])
        with pytest.raises(pybamm.OptionError, match=r"1D array"):
            solution.save_vtu(path, ["Both regions [K]"], t=[[0.0, 1.0]])
        with pytest.raises(pybamm.OptionError, match=r"1D array"):
            solution.save_vtu(path, ["Both regions [K]"], t=[])
        assert not path.exists()

    def test_rejects_structured_variables(self, tmp_path):
        model = pybamm.BaseModel()
        x = pybamm.SpatialVariable("x", domain="line")
        model._geometry = {
            "line": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        line = pybamm.StateVector(slice(0, 2), domain="line").with_mesh(
            pybamm.SubMesh1D(np.array([0.0, 0.5, 1.0]), "cartesian")
        )
        model.variables = {"line": line}
        model.update_processed_variables(model.variables)
        solution = pybamm.Solution(
            np.array([0.0, 1.0]), np.asfortranarray([[1.0, 2.0], [3.0, 4.0]]), model, {}
        )
        with pytest.raises(pybamm.OptionError, match=r"Cannot export 'line'"):
            solution.save_vtu(tmp_path / "out", ["line"])

    def test_rejects_incompatible_meshes(self, tmp_path):
        quad = _strip_mesh(0.0, 1.0, 1, 2)
        hexahedron = _strip_mesh(0.0, 1.0, 1, 3)
        triangle = pybamm.UnstructuredSubMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
        )
        model = pybamm.BaseModel()
        model._geometry = {
            **_geometry(["quad", "triangle"], 2),
            **_geometry(["hexahedron"], 3),
        }
        model.variables = {
            name: pybamm.StateVector(slice(0, 1), domain=name).with_mesh(mesh)
            for name, mesh in [
                ("quad", quad),
                ("hexahedron", hexahedron),
                ("triangle", triangle),
            ]
        }
        model.update_processed_variables(model.variables)
        solution = pybamm.Solution(
            np.array([0.0, 1.0]), np.asfortranarray([[1.0, 2.0]]), model, {}
        )
        with pytest.raises(pybamm.GeometryError, match=r"different dimensions"):
            solution.save_vtu(tmp_path / "out", ["quad", "hexahedron"])
        with pytest.raises(pybamm.GeometryError, match=r"different element types"):
            solution.save_vtu(tmp_path / "out", ["quad", "triangle"])

    def test_write_failure_raises(self, tmp_path):
        solution, _ = _two_region_solution(2)
        # a directory where the .vtu should go makes VTK's writer fail
        (tmp_path / "out" / "out_0000.vtu").mkdir(parents=True)
        with pytest.raises(OSError, match=r"VTK failed to write"):
            solution.save_vtu(tmp_path / "out", ["Both regions [K]"], t=[0.0])
