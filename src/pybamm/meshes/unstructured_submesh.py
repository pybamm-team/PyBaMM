import numpy as np

import pybamm

from .meshes import MeshGenerator, SubMesh


class UnstructuredSubMesh(SubMesh):
    """
    Cell-centered finite volume submesh on polygonal/polyhedral elements.

    Supported element types:

    * **2D**: triangles (3 vertices) or quadrilaterals (4 vertices)
    * **3D**: tetrahedra (4 vertices)

    All operators are dimension-agnostic: the same code path handles
    both 2D and 3D, with dimension inferred from ``nodes.shape[1]``.

    Parameters
    ----------
    nodes : numpy.ndarray, shape (n_nodes, d)
        Vertex coordinates (d = 2 or 3).
    elements : numpy.ndarray, shape (n_cells, n_verts_per_cell)
        Element vertex indices.  For 2D: 3 (triangles) or 4 (quads).
        For 3D: 4 (tetrahedra).
    coord_sys : str, optional
        Coordinate system, default ``"cartesian"``.
    boundary_faces : dict[str, numpy.ndarray] or None, optional
        Maps boundary name to face indices.  If ``None``, boundaries
        are auto-detected from face centroid positions.
    """

    def __init__(self, nodes, elements, coord_sys="cartesian", boundary_faces=None):
        super().__init__()
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.dimension = self.nodes.shape[1]
        self.coord_sys = coord_sys

        verts_per_cell = self.elements.shape[1]
        if self.dimension == 2 and verts_per_cell == 4:
            self.element_type = "quad"
        elif self.dimension == 2 and verts_per_cell == 3:
            self.element_type = "triangle"
        elif self.dimension == 3 and verts_per_cell == 4:
            self.element_type = "tetrahedron"
        else:
            raise ValueError(
                f"Unsupported: {verts_per_cell} vertices per cell in {self.dimension}D"
            )

        self._compute_cell_geometry()
        self._build_face_connectivity()
        self._compute_face_geometry()

        if boundary_faces is not None:
            self.boundary_faces = boundary_faces
        else:
            self._identify_boundary_faces()

        self.npts = len(self.elements)
        self.internal_boundaries = []
        self.interface_data = {}

    # ------------------------------------------------------------------
    # Cell geometry
    # ------------------------------------------------------------------

    def _compute_cell_geometry(self):
        verts = self.nodes[self.elements]  # (n_cells, n_verts, d)
        self.cell_centroids = verts.mean(axis=1)

        if self.element_type == "triangle":
            v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
            cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (
                v1[:, 1] - v0[:, 1]
            ) * (v2[:, 0] - v0[:, 0])
            self.cell_volumes = 0.5 * np.abs(cross)
        elif self.element_type == "quad":
            # Shoelace formula for arbitrary (convex) quadrilaterals
            # Vertices ordered: v0, v1, v2, v3 (counterclockwise or clockwise)
            x = verts[:, :, 0]  # (n_cells, 4)
            y = verts[:, :, 1]  # (n_cells, 4)
            # shoelace: sum_i (x_i * y_{i+1} - x_{i+1} * y_i)
            x_next = np.roll(x, -1, axis=1)
            y_next = np.roll(y, -1, axis=1)
            self.cell_volumes = 0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1))
        elif self.element_type == "tetrahedron":
            v0, v1, v2, v3 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
            d1 = v1 - v0
            d2 = v2 - v0
            d3 = v3 - v0
            det = (
                d1[:, 0] * (d2[:, 1] * d3[:, 2] - d2[:, 2] * d3[:, 1])
                - d1[:, 1] * (d2[:, 0] * d3[:, 2] - d2[:, 2] * d3[:, 0])
                + d1[:, 2] * (d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0])
            )
            self.cell_volumes = np.abs(det) / 6.0

    # ------------------------------------------------------------------
    # Face-cell connectivity
    # ------------------------------------------------------------------

    def _build_face_connectivity(self):
        """Extract faces, identify internal / boundary, record owner-neighbor."""
        d = self.dimension
        n_verts_per_face = d  # edges (2 verts) in 2D, triangles (3 verts) in 3D

        face_dict = {}  # canonical key -> owner_cell

        internal_owner = []
        internal_neighbor = []
        internal_face_verts = []

        for cell_idx, cell_verts in enumerate(self.elements):
            for face_verts in self._cell_faces(cell_verts):
                key = tuple(sorted(face_verts))

                if key in face_dict:
                    other_cell = face_dict.pop(key)
                    internal_owner.append(other_cell)
                    internal_neighbor.append(cell_idx)
                    internal_face_verts.append(key)
                else:
                    face_dict[key] = cell_idx

        # Remaining entries are boundary faces
        boundary_owner_list = []
        boundary_face_verts = []
        for key, cell_idx in face_dict.items():
            boundary_owner_list.append(cell_idx)
            boundary_face_verts.append(key)

        n_internal = len(internal_owner)
        n_boundary = len(boundary_owner_list)

        all_face_verts = internal_face_verts + boundary_face_verts
        all_owner = internal_owner + boundary_owner_list

        self.faces = np.array(all_face_verts, dtype=int).reshape(-1, n_verts_per_face)
        self.face_owner = np.array(all_owner, dtype=int)
        self.face_neighbor = np.array(internal_neighbor, dtype=int)
        self.n_internal_faces = n_internal
        self._n_boundary_faces = n_boundary
        self._boundary_face_start = n_internal

    def _cell_faces(self, cell_verts):
        """Yield face vertex tuples for a single cell."""
        n = len(cell_verts)
        if self.element_type == "quad":
            # 4 edges: (v0,v1), (v1,v2), (v2,v3), (v3,v0)
            for i in range(n):
                yield (cell_verts[i], cell_verts[(i + 1) % n])
        else:
            # Simplex: d+1 faces, face i omits vertex i
            for skip in range(n):
                yield tuple(cell_verts[j] for j in range(n) if j != skip)

    # ------------------------------------------------------------------
    # Face geometry
    # ------------------------------------------------------------------

    def _compute_face_geometry(self):
        face_verts = self.nodes[self.faces]  # (n_faces, d, d)

        self.face_centroids = face_verts.mean(axis=1)

        if self.dimension == 2:
            # Face = edge: 2 vertices
            v0, v1 = face_verts[:, 0], face_verts[:, 1]
            edge = v1 - v0
            self.face_areas = np.linalg.norm(edge, axis=1)
            # Outward normal: perpendicular to edge (rotate 90 degrees)
            normals = np.column_stack([edge[:, 1], -edge[:, 0]])
        else:
            # Face = triangle: 3 vertices
            v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
            cross = np.cross(v1 - v0, v2 - v0)
            self.face_areas = 0.5 * np.linalg.norm(cross, axis=1)
            normals = cross  # will be normalized below

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-30, 1.0, norms)
        normals = normals / norms

        # Orient outward from owner cell: if the normal points from the
        # owner centroid toward the face centroid, keep it; otherwise flip.
        owner_centroids = self.cell_centroids[self.face_owner]
        to_face = self.face_centroids - owner_centroids
        dot = np.sum(normals * to_face, axis=1)
        flip = dot < 0
        normals[flip] *= -1

        self.face_normals = normals

    # ------------------------------------------------------------------
    # Boundary identification
    # ------------------------------------------------------------------

    def _identify_boundary_faces(self):
        bnd_start = self._boundary_face_start
        bnd_centroids = self.face_centroids[bnd_start:]

        if len(bnd_centroids) == 0:
            self.boundary_faces = {}
            return

        tol = 1e-10
        x_min, x_max = bnd_centroids[:, 0].min(), bnd_centroids[:, 0].max()

        tag_map = {
            "left": np.abs(bnd_centroids[:, 0] - x_min) < tol,
            "right": np.abs(bnd_centroids[:, 0] - x_max) < tol,
        }

        if self.dimension >= 2:
            z_col = 1 if self.dimension == 2 else 2
            z_min = bnd_centroids[:, z_col].min()
            z_max = bnd_centroids[:, z_col].max()
            tag_map["bottom"] = np.abs(bnd_centroids[:, z_col] - z_min) < tol
            tag_map["top"] = np.abs(bnd_centroids[:, z_col] - z_max) < tol

        if self.dimension == 3:
            y_min, y_max = bnd_centroids[:, 1].min(), bnd_centroids[:, 1].max()
            tag_map["front"] = np.abs(bnd_centroids[:, 1] - y_min) < tol
            tag_map["back"] = np.abs(bnd_centroids[:, 1] - y_max) < tol

        self.boundary_faces = {}
        for name, mask in tag_map.items():
            indices = np.nonzero(mask)[0] + bnd_start
            if len(indices) > 0:
                self.boundary_faces[name] = indices


# ======================================================================
# Mesh generators
# ======================================================================


class UnstructuredMeshGenerator(MeshGenerator):
    """
    Built-in generator that creates meshes from structured grids.

    * **2D**: rectangular domain meshed as quads or triangulated by
      splitting each quad into 2 triangles.
    * **3D**: rectangular prism meshed by splitting each hex into 5
      tetrahedra.

    Parameters
    ----------
    coord_sys : str, optional
        Coordinate system, default ``"cartesian"``.
    element_type : str, optional
        ``"quad"`` for quadrilateral cells (2D only, TPFA-orthogonal),
        ``"triangle"`` for triangular cells (2D default),
        ``"tetrahedron"`` for tetrahedral cells (3D default).
        If ``None``, defaults to ``"triangle"`` in 2D and
        ``"tetrahedron"`` in 3D.
    """

    def __init__(self, coord_sys="cartesian", element_type=None):
        self.submesh_type = UnstructuredSubMesh
        self.submesh_params = {}
        self.coord_sys = coord_sys
        self._element_type = element_type

    def __call__(self, lims, npts):
        spatial_vars, spatial_lims = self._parse_lims(lims)
        dim = len(spatial_vars)
        if dim == 2:
            return self._generate_2d(spatial_vars, spatial_lims, npts)
        elif dim == 3:
            return self._generate_3d(spatial_vars, spatial_lims, npts)
        else:
            raise ValueError(
                f"UnstructuredMeshGenerator supports 2D and 3D, got {dim} spatial variables"
            )

    def __repr__(self):
        return "Generator for UnstructuredSubMesh"

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_lims(lims):
        spatial_vars = []
        spatial_lims = []
        for var, var_lims in lims.items():
            if var == "tabs":
                continue
            if isinstance(var, str):
                var = getattr(pybamm.standard_spatial_vars, var)
            spatial_vars.append(var)
            spatial_lims.append(var_lims)
        return spatial_vars, spatial_lims

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    def _generate_2d(self, spatial_vars, spatial_lims, npts):
        var_x, var_z = spatial_vars
        lim_x, lim_z = spatial_lims
        nx = npts[var_x.name]
        nz = npts[var_z.name]

        x_edges = np.linspace(lim_x["min"], lim_x["max"], nx + 1)
        z_edges = np.linspace(lim_z["min"], lim_z["max"], nz + 1)

        etype = self._element_type or "triangle"
        if etype == "quad":
            nodes, elements = _make_quad_grid(x_edges, z_edges)
        elif etype == "triangle":
            nodes, elements = _quad_to_tri(x_edges, z_edges)
        else:
            raise ValueError(f"Unsupported 2D element_type: {etype!r}")
        return UnstructuredSubMesh(nodes, elements, coord_sys=self.coord_sys)

    # ------------------------------------------------------------------
    # 3D: hex -> 5 tets
    # ------------------------------------------------------------------

    def _generate_3d(self, spatial_vars, spatial_lims, npts):
        var_x, var_y, var_z = spatial_vars
        lim_x, lim_y, lim_z = spatial_lims
        nx = npts[var_x.name]
        ny = npts[var_y.name]
        nz = npts[var_z.name]

        x_edges = np.linspace(lim_x["min"], lim_x["max"], nx + 1)
        y_edges = np.linspace(lim_y["min"], lim_y["max"], ny + 1)
        z_edges = np.linspace(lim_z["min"], lim_z["max"], nz + 1)

        nodes, elements = _hex_to_tet(x_edges, y_edges, z_edges)
        return UnstructuredSubMesh(nodes, elements, coord_sys=self.coord_sys)


class UserSuppliedUnstructuredMesh(MeshGenerator):
    """
    Load a simplex mesh from an external file via *meshio*.

    Parameters
    ----------
    filepath : str
        Path to the mesh file (GMSH ``.msh``, VTK ``.vtu``, etc.).
    subdomain_mapping : dict[str, int] or None
        Maps PyBaMM domain name to physical group / cell-data tag.
    boundary_mapping : dict[str, int] or None
        Maps boundary name to physical group / facet tag.
    coord_sys : str, optional
        Coordinate system, default ``"cartesian"``.
    """

    def __init__(
        self,
        filepath,
        subdomain_mapping=None,
        boundary_mapping=None,
        coord_sys="cartesian",
    ):
        self.submesh_type = UnstructuredSubMesh
        self.submesh_params = {}
        self.filepath = filepath
        self.subdomain_mapping = subdomain_mapping or {}
        self.boundary_mapping = boundary_mapping or {}
        self.coord_sys = coord_sys
        self._cached_mesh = None

    def __call__(self, lims, npts):
        import meshio

        if self._cached_mesh is None:
            self._cached_mesh = meshio.read(self.filepath)

        mesh = self._cached_mesh
        nodes = mesh.points

        # Determine which domain is being requested from the lims keys
        domain_name = self._domain_name_from_lims(lims)

        # Extract simplex cells (triangles or tets)
        simplex_cells, simplex_type = self._extract_simplex_cells(mesh)

        if domain_name and domain_name in self.subdomain_mapping:
            tag_value = self.subdomain_mapping[domain_name]
            cell_mask = self._get_cell_mask(mesh, simplex_type, tag_value)
            elements = simplex_cells[cell_mask]
        else:
            elements = simplex_cells

        # Re-index nodes to compact numbering
        unique_nodes = np.unique(elements)
        node_map = np.full(nodes.shape[0], -1, dtype=int)
        node_map[unique_nodes] = np.arange(len(unique_nodes))
        compact_nodes = nodes[unique_nodes]
        compact_elements = node_map[elements]

        # Trim to 2D if all z-coordinates are zero
        if compact_nodes.shape[1] == 3 and np.allclose(compact_nodes[:, 2], 0):
            compact_nodes = compact_nodes[:, :2]

        return UnstructuredSubMesh(
            compact_nodes, compact_elements, coord_sys=self.coord_sys
        )

    def __repr__(self):
        return f"UserSuppliedUnstructuredMesh({self.filepath})"

    @staticmethod
    def _domain_name_from_lims(lims):
        for var in lims:
            if var == "tabs":
                continue
            if isinstance(var, str):
                name = var
            else:
                name = var.name
            for prefix in ("x_n", "x_s", "x_p"):
                if name.startswith(prefix):
                    domain_map = {
                        "x_n": "negative electrode",
                        "x_s": "separator",
                        "x_p": "positive electrode",
                    }
                    return domain_map.get(prefix)
        return None

    @staticmethod
    def _extract_simplex_cells(mesh):
        for block in mesh.cells:
            if block.type == "tetra":
                return block.data, "tetra"
            if block.type == "triangle":
                return block.data, "triangle"
        raise ValueError("No simplex cells (triangle or tetra) found in mesh file")

    @staticmethod
    def _get_cell_mask(mesh, cell_type, tag_value):
        for _key, data_list in mesh.cell_data.items():
            for block, data in zip(mesh.cells, data_list, strict=False):
                if block.type == cell_type:
                    return data == tag_value
        raise ValueError(
            f"Could not find cell data tag {tag_value} for cell type '{cell_type}'"
        )


# ======================================================================
# Interface data
# ======================================================================


def compute_interface_data(left_mesh, right_mesh, left_name=None, right_name=None):
    """
    Compute coupling data for the interface between two adjacent
    :class:`UnstructuredSubMesh` objects.

    Finds "right" boundary faces of *left_mesh* and "left" boundary faces
    of *right_mesh*, matches them by face centroid position, and records
    cell indices, face areas, and centroid-to-centroid distances.

    Parameters
    ----------
    left_mesh : UnstructuredSubMesh
    right_mesh : UnstructuredSubMesh
    left_name : str or None
        Domain name of the left mesh (stored as key in ``interface_data``).
    right_name : str or None
        Domain name of the right mesh (stored as key in ``interface_data``).

    Returns
    -------
    dict
        Keys: ``"left_cells"``, ``"right_cells"``, ``"face_areas"``,
        ``"cell_distances"``.
    """
    left_bnd = left_mesh.boundary_faces.get("right", np.array([], dtype=int))
    right_bnd = right_mesh.boundary_faces.get("left", np.array([], dtype=int))

    if len(left_bnd) == 0 or len(right_bnd) == 0:
        raise ValueError(
            "Cannot compute interface data: one or both meshes have no "
            "matching boundary faces ('right' on left_mesh, 'left' on right_mesh)."
        )

    left_centroids = left_mesh.face_centroids[left_bnd]
    right_centroids = right_mesh.face_centroids[right_bnd]

    # Match faces by transverse coordinates (all coords except x)
    left_transverse = left_centroids[:, 1:]
    right_transverse = right_centroids[:, 1:]

    # Build a mapping by closest transverse match
    from scipy.spatial import cKDTree

    tree = cKDTree(right_transverse)
    dists, right_indices = tree.query(left_transverse)

    tol = 1e-8 * max(
        np.ptp(left_transverse, axis=0).max(),
        np.ptp(right_transverse, axis=0).max(),
        1.0,
    )
    if np.any(dists > tol):
        raise ValueError(
            f"Interface faces do not match: max transverse mismatch = {dists.max():.2e}. "
            "Ensure both meshes have the same transverse grid."
        )

    left_cells = left_mesh.face_owner[left_bnd]
    right_cells = right_mesh.face_owner[right_bnd[right_indices]]
    face_areas = left_mesh.face_areas[left_bnd]

    left_cell_centroids = left_mesh.cell_centroids[left_cells]
    right_cell_centroids = right_mesh.cell_centroids[right_cells]
    cell_distances = np.linalg.norm(right_cell_centroids - left_cell_centroids, axis=1)

    result = {
        "left_cells": left_cells,
        "right_cells": right_cells,
        "face_areas": face_areas,
        "cell_distances": cell_distances,
    }

    if right_name is not None:
        left_mesh.interface_data[right_name] = result
    if left_name is not None:
        right_mesh.interface_data[left_name] = {
            "left_cells": right_cells,
            "right_cells": left_cells,
            "face_areas": face_areas,
            "cell_distances": cell_distances,
        }

    return result


# ======================================================================
# Grid-to-simplex helpers
# ======================================================================


def _make_quad_grid(x_edges, z_edges):
    """
    Build a structured quadrilateral mesh on a rectangle.

    Vertices are ordered counterclockwise so that the shoelace formula
    gives a positive area and consecutive-edge face enumeration is
    consistent.

    Returns
    -------
    nodes : (n_nodes, 2)
    elements : (n_cells, 4)
    """
    nx = len(x_edges) - 1
    nz = len(z_edges) - 1
    xx, zz = np.meshgrid(x_edges, z_edges, indexing="ij")
    nodes = np.column_stack([xx.ravel(), zz.ravel()])

    def node_id(i, j):
        return i * (nz + 1) + j

    elements = []
    for i in range(nx):
        for j in range(nz):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i + 1, j + 1)
            n3 = node_id(i, j + 1)
            elements.append([n0, n1, n2, n3])

    return nodes, np.array(elements, dtype=int)


def _quad_to_tri(x_edges, z_edges):
    """
    Triangulate a rectangle defined by ``x_edges`` and ``z_edges``.

    Each quad cell is split into 2 triangles using the lower-left to
    upper-right diagonal (consistent across all cells for interface
    conformity).

    Returns
    -------
    nodes : (n_nodes, 2)
    elements : (n_cells, 3)
    """
    nx = len(x_edges) - 1
    nz = len(z_edges) - 1
    xx, zz = np.meshgrid(x_edges, z_edges, indexing="ij")
    nodes = np.column_stack([xx.ravel(), zz.ravel()])

    def node_id(i, j):
        return i * (nz + 1) + j

    elements = []
    for i in range(nx):
        for j in range(nz):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i + 1, j + 1)
            n3 = node_id(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    return nodes, np.array(elements, dtype=int)


def _hex_to_tet(x_edges, y_edges, z_edges):
    """
    Tetrahedralise a rectangular prism defined by edge arrays.

    Each hex cell is split into 5 tetrahedra using a consistent
    decomposition that guarantees matching triangular faces on
    axis-aligned planes (required for interface conformity).

    The decomposition alternates orientation based on the parity of
    (i + j + k) so that shared faces between adjacent hexes are
    triangulated identically.

    Returns
    -------
    nodes : (n_nodes, 3)
    elements : (n_cells, 4)
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1

    xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def node_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    # Two 5-tet decomposition patterns that share identical face diagonals
    # on every axis-aligned interface.
    # Hex vertices numbered:
    #   0 = (i,   j,   k  )   4 = (i,   j,   k+1)
    #   1 = (i+1, j,   k  )   5 = (i+1, j,   k+1)
    #   2 = (i+1, j+1, k  )   6 = (i+1, j+1, k+1)
    #   3 = (i,   j+1, k  )   7 = (i,   j+1, k+1)
    #
    # Pattern A (even parity): diagonal from vertex 0 to 6
    pattern_a = [
        (0, 1, 2, 5),
        (0, 2, 3, 7),
        (0, 5, 7, 4),
        (2, 5, 7, 6),
        (0, 2, 5, 7),
    ]
    # Pattern B (odd parity): diagonal from vertex 1 to 7
    pattern_b = [
        (1, 0, 3, 4),
        (1, 2, 3, 6),
        (1, 6, 4, 5),
        (3, 4, 6, 7),
        (1, 3, 4, 6),
    ]

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                hex_verts = [
                    node_id(i, j, k),
                    node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k),
                    node_id(i, j + 1, k),
                    node_id(i, j, k + 1),
                    node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i, j + 1, k + 1),
                ]
                pattern = pattern_a if (i + j + k) % 2 == 0 else pattern_b
                for tet in pattern:
                    elements.append([hex_verts[v] for v in tet])

    return nodes, np.array(elements, dtype=int)
