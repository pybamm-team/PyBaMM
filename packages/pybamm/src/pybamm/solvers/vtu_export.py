"""
Export of unstructured-mesh solutions to VTK files (``.vtu`` + ``.pvd``).
"""

from __future__ import annotations

import os
from xml.sax.saxutils import escape, quoteattr  # nosec B406 - escaping only

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.meshes.unstructured_submesh import _geometric_tolerance
from pybamm.plotting.plot_vtk import _build_vtk_grid, _set_cell_scalars

# output times evaluated together; bounds memory to cells x chunk per variable
_TIME_CHUNK_SIZE = 64


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


def _match(existing, points, tolerance):
    """Map ``points`` onto ``existing`` points within ``tolerance``, appending the rest.

    Returns the index of each point in the extended array and a mask of the
    points that were appended.
    """
    from scipy.spatial import cKDTree

    if len(existing) == 0:
        return np.arange(len(points)), np.ones(len(points), dtype=bool)
    distance, nearest = cKDTree(existing).query(points)
    is_new = distance >= tolerance
    return np.where(is_new, len(existing) + np.cumsum(is_new) - 1, nearest), is_new


class _UnionMesh:
    """Union of the cells of several unstructured meshes.

    Cells are matched across meshes by centroid, so a cell shared by a
    multi-region mesh and a single-region mesh is written once. Assumes any
    two meshes either share a cell exactly or do not overlap there. Provides
    the ``vertices``/``elements``/``element_type`` of a mesh.

    Parameters
    ----------
    meshes : iterable of :class:`pybamm.UnstructuredSubMesh`
        The meshes of the exported variables.
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

        vertices = np.empty((0, meshes[0].dimension))
        centroids = np.empty((0, meshes[0].dimension))
        elements = []
        self._cell_maps = {}
        for mesh in meshes:
            cell_map, is_new = _match(centroids, mesh.cell_centroids, tolerance)
            self._cell_maps[id(mesh)] = cell_map
            if not is_new.any():
                continue
            centroids = np.vstack([centroids, mesh.cell_centroids[is_new]])
            new_elements = mesh.elements[is_new]
            used, connectivity = np.unique(new_elements, return_inverse=True)
            # weld vertices shared with earlier meshes, so interfaces stay connected
            vertex_map, vertex_is_new = _match(vertices, mesh.vertices[used], tolerance)
            vertices = np.vstack([vertices, mesh.vertices[used][vertex_is_new]])
            elements.append(vertex_map[connectivity].reshape(new_elements.shape))
        self.vertices = vertices
        self.elements = np.vstack(elements)
        self.npts = len(centroids)

    def scatter(self, mesh, values):
        """Place per-cell ``values`` of ``mesh`` on the union cells, NaN elsewhere."""
        out = np.full((self.npts, *values.shape[1:]), np.nan)
        out[self._cell_maps[id(mesh)]] = values
        return out


def _evaluate(union, kind, processed_variable, times):
    """Values of a variable on the union cells: ``(cells, [3,] times)``."""
    data = processed_variable(t=times)
    if kind == "vector":
        data = np.stack(data, axis=1)
        # VTK vectors have 3 components; the third is zero for 2D meshes
        padding = np.zeros((data.shape[0], 3 - data.shape[1], data.shape[2]))
        data = np.concatenate([data, padding], axis=1)
    return union.scatter(processed_variable.mesh, data)


def _vtk_array_name(name):
    """Escape a variable name for VTK, whose XML writer writes array names raw."""
    return escape(name, {'"': "&quot;"})


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
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<VTKFile type="Collection" version="0.1">',
        "  <Collection>",
    ]
    lines += [
        f'    <DataSet timestep="{time!r}" part="0" file={quoteattr(path)}/>'
        for time, path in entries
    ]
    lines += ["  </Collection>", "</VTKFile>", ""]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_vtu(
    solution: pybamm.Solution,
    filename: str | os.PathLike,
    variables: str | list[str],
    t: npt.ArrayLike | None = None,
) -> str:
    """Write unstructured-mesh variables to one ``.vtu`` per time plus a ``.pvd``.

    See :meth:`pybamm.Solution.save_vtu` for the output layout.

    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        The solution to export.
    filename : str or os.PathLike
        Path of the ``.pvd`` file; ``.pvd`` is appended if missing.
    variables : str or list of str
        Names of the variables to export.
    t : array-like, optional
        Strictly increasing output times [s]. Defaults to the solution's times.

    Returns
    -------
    str
        Path of the written ``.pvd`` file.
    """
    vtk = pybamm.import_optional_dependency("vtk")

    filename = os.fspath(filename)
    if not filename.endswith(".pvd"):
        filename += ".pvd"
    directory, basename = os.path.split(filename)
    stem = basename[: -len(".pvd")]
    if not stem:
        raise pybamm.OptionError(
            f"filename {filename!r} has no file name to name the output after; "
            "pass a path such as 'output/solution.pvd'."
        )

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
        if not np.all(np.isfinite(times)):
            raise pybamm.OptionError("t must contain only finite times.")
        if np.any(np.diff(times) <= 0):
            raise pybamm.OptionError("t must be strictly increasing.")
        t_min, t_max = solution.t[0], solution.t[-1]
        # absorb rounding in user-built times, e.g. hours * 3600
        tolerance = 1e-10 * max(abs(t_min), abs(t_max), 1.0)
        if np.any((times < t_min - tolerance) | (times > t_max + tolerance)):
            raise pybamm.OptionError(
                "Output times must lie within the solution's time range "
                f"[{t_min}, {t_max}] s."
            )
        times = np.clip(times, t_min, t_max)
        if np.any(np.diff(times) <= 0):
            raise pybamm.OptionError(
                "t has several times within rounding of the same end of the "
                f"solution's time range [{t_min}, {t_max}] s."
            )

    spatial, scalars = [], []
    for name in variables:
        processed_variable = solution[name]
        kind = _classify(name, processed_variable)
        if kind == "scalar":
            scalars.append(
                (_vtk_array_name(name), np.ravel(processed_variable(t=times)))
            )
        else:
            spatial.append((_vtk_array_name(name), kind, processed_variable))
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

    os.makedirs(os.path.join(directory, stem), exist_ok=True)

    grid = _build_vtk_grid(union)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(grid)
    width = max(4, len(str(len(times) - 1)))
    entries = []
    for start in range(0, len(times), _TIME_CHUNK_SIZE):
        chunk = times[start : start + _TIME_CHUNK_SIZE]
        cell_values = [
            (name, kind, _evaluate(union, kind, processed_variable, chunk))
            for name, kind, processed_variable in spatial
        ]
        for j, time in enumerate(chunk):
            i = start + j
            for name, kind, values in cell_values:
                if kind == "vector":
                    _set_cell_vectors(grid, name, values[..., j])
                else:
                    _set_cell_scalars(grid, name, values[:, j])
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
