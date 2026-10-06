"""
Export of unstructured-mesh solutions to VTK files (``.vtu`` + ``.pvd``).
"""

from __future__ import annotations

import os
from xml.sax.saxutils import quoteattr

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.meshes.unstructured_submesh import _geometric_tolerance
from pybamm.plotting.plot_vtk import _build_vtk_grid, _set_cell_scalars


def _classify(name, processed_variable):
    """Return ``"cell"``, ``"vector"`` or ``"scalar"`` for an exportable variable."""
    if isinstance(processed_variable, pybamm.ProcessedVariableUnstructuredFVM):
        return "cell"
    if isinstance(
        processed_variable, pybamm.ProcessedVariableVectorFieldUnstructuredFVM
    ):
        return "vector"
    if getattr(processed_variable, "dimensions", None) == 0:
        return "scalar"
    raise pybamm.OptionError(
        f"Cannot export '{name}' to VTU: only cell-centred variables on an "
        "unstructured finite-volume mesh (scalar or vector field) and 0D "
        f"variables are supported, but it is a {type(processed_variable).__name__}."
    )


class _UnionMesh:
    """Union of the cells of several unstructured meshes.

    Meshes of different variables overlap: a variable over three regions lives
    on the welded mesh of those regions, which repeats the cells of a
    single-region variable. Cells are matched across meshes by centroid, so
    every cell is written once and each variable maps onto the cells it covers.
    Provides the ``vertices``/``elements``/``element_type`` of a mesh.
    """

    def __init__(self, meshes):
        meshes = sorted(meshes, key=lambda mesh: mesh.npts, reverse=True)
        element_types = {mesh.element_type for mesh in meshes}
        if len(element_types) > 1:
            raise pybamm.GeometryError(
                "Cannot export variables on meshes of different element types "
                f"({sorted(t.value for t in element_types)}) to one VTU file."
            )
        self.element_type = meshes[0].element_type
        tolerance = _geometric_tolerance(meshes)

        vertices, elements, centroids = [], [], []
        self.npts = 0
        n_vertices = 0
        self._cell_maps = {}
        for mesh in meshes:
            if self.npts == 0:
                is_new = np.ones(mesh.npts, dtype=bool)
                cell_map = np.arange(mesh.npts)
            else:
                from scipy.spatial import cKDTree

                tree = cKDTree(np.vstack(centroids))
                distance, nearest = tree.query(mesh.cell_centroids)
                is_new = distance >= tolerance
                cell_map = np.where(is_new, self.npts + np.cumsum(is_new) - 1, nearest)
            self._cell_maps[id(mesh)] = cell_map
            if not is_new.any():
                continue
            new_elements = mesh.elements[is_new]
            used, connectivity = np.unique(new_elements, return_inverse=True)
            vertices.append(mesh.vertices[used])
            elements.append(connectivity.reshape(new_elements.shape) + n_vertices)
            centroids.append(mesh.cell_centroids[is_new])
            n_vertices += len(used)
            self.npts += int(is_new.sum())
        self.vertices = np.vstack(vertices)
        self.elements = np.vstack(elements)

    def scatter(self, mesh, values):
        """Place per-cell ``values`` of ``mesh`` on the union cells, NaN elsewhere."""
        out = np.full((self.npts, *values.shape[1:]), np.nan)
        out[self._cell_maps[id(mesh)]] = values
        return out


def _set_field_data(grid, name, values):
    """Set (or update) a named float array on a grid's field data."""
    numpy_support = pybamm.import_optional_dependency("vtk.util.numpy_support")

    field_data = grid.GetFieldData()
    field_data.RemoveArray(name)
    array = numpy_support.numpy_to_vtk(np.atleast_1d(values).astype(float), deep=True)
    array.SetName(name)
    field_data.AddArray(array)


def _set_cell_vectors(grid, name, values):
    """Set (or replace) a 3-component cell array on a grid."""
    numpy_support = pybamm.import_optional_dependency("vtk.util.numpy_support")

    cell_data = grid.GetCellData()
    cell_data.RemoveArray(name)
    array = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(values, dtype=np.float64), deep=True
    )
    array.SetName(name)
    cell_data.AddArray(array)
    grid.Modified()


def _write_pvd(filename, entries):
    """Write a ParaView collection file indexing ``(time, path)`` entries."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1">',
        "  <Collection>",
    ]
    lines += [
        f'    <DataSet timestep="{time!r}" part="0" file={quoteattr(path)}/>'
        for time, path in entries
    ]
    lines += ["  </Collection>", "</VTKFile>", ""]
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def save_vtu(
    solution: pybamm.Solution,
    filename: str | os.PathLike,
    variables: str | list[str],
    t: npt.ArrayLike | None = None,
) -> str:
    """Write unstructured-mesh variables to one ``.vtu`` per time plus a ``.pvd``.

    See :meth:`pybamm.Solution.save_vtu`.
    """
    vtk = pybamm.import_optional_dependency("vtk")

    if isinstance(variables, str):
        variables = [variables]
    if len(variables) == 0:
        raise pybamm.OptionError("save_vtu needs at least one variable to export.")

    if t is None:
        times = np.unique(solution.t)
    else:
        times = np.asarray(t, dtype=float)
        if times.ndim > 1 or times.size == 0:
            raise pybamm.OptionError("t must be a scalar or a non-empty 1D array.")
        times = np.atleast_1d(times)
        t_min, t_max = solution.t[0], solution.t[-1]
        if np.any((times < t_min) | (times > t_max)):
            raise pybamm.OptionError(
                "Output times must lie within the solution's time range "
                f"[{t_min}, {t_max}] s."
            )

    spatial, scalars = [], []
    for name in variables:
        processed_variable = solution[name]
        kind = _classify(name, processed_variable)
        if kind == "scalar":
            scalars.append((name, np.ravel(processed_variable(t=times))))
        else:
            spatial.append((name, kind, processed_variable))
    if not spatial:
        raise pybamm.OptionError(
            "save_vtu needs at least one variable on an unstructured mesh; 0D "
            "variables are written alongside them as field data."
        )

    meshes = {id(pv.mesh): pv.mesh for _, _, pv in spatial}
    dimensions = {mesh.dimension for mesh in meshes.values()}
    if len(dimensions) > 1:
        raise pybamm.GeometryError(
            "Cannot export variables on meshes of different dimensions to one "
            f"VTU file (got dimensions {sorted(dimensions)})."
        )
    union = _UnionMesh(meshes.values())

    # evaluate each variable once for all output times: (cells, [components,] times)
    cell_values = {}
    for name, kind, processed_variable in spatial:
        data = processed_variable(t=times)
        if kind == "vector":
            data = np.stack(data, axis=1)
            # VTK vectors have 3 components; 2D (x, z) vectors lie in the z = 0 plane
            padding = np.zeros((data.shape[0], 3 - data.shape[1], data.shape[2]))
            data = np.concatenate([data, padding], axis=1)
        cell_values[name] = (kind, union.scatter(processed_variable.mesh, data))

    filename = os.fspath(filename)
    if not filename.endswith(".pvd"):
        filename += ".pvd"
    directory, basename = os.path.split(filename)
    stem = basename[: -len(".pvd")]
    os.makedirs(os.path.join(directory, stem), exist_ok=True)

    grid = _build_vtk_grid(union)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(grid)
    width = max(4, len(str(len(times) - 1)))
    entries = []
    for i, time in enumerate(times):
        for name, (kind, values) in cell_values.items():
            if kind == "vector":
                _set_cell_vectors(grid, name, values[..., i])
            else:
                _set_cell_scalars(grid, name, values[:, i])
        # ParaView reads the "TimeValue" field as the dataset's time
        _set_field_data(grid, "TimeValue", time)
        for name, values in scalars:
            _set_field_data(grid, name, values[i])

        # the .pvd stores paths relative to itself, so the output can be moved
        relative_path = f"{stem}/{stem}_{i:0{width}d}.vtu"
        writer.SetFileName(os.path.join(directory, relative_path))
        # some VTK versions return 1 on failure and only set the error code
        if writer.Write() != 1 or writer.GetErrorCode() != 0:
            raise OSError(f"VTK failed to write {writer.GetFileName()!r}")
        entries.append((float(time), relative_path))

    _write_pvd(filename, entries)
    return filename
