from enum import Enum

import numpy as np

import pybamm

from .meshes import MeshGenerator, SubMesh


class ElementType(str, Enum):
    """Element types supported by :class:`UnstructuredSubMesh`.

    The ``str`` mixin keeps plain strings (``element_type="quad"``)
    working everywhere an ``ElementType`` is expected.
    """

    TRIANGLE = "triangle"
    QUAD = "quad"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"

    @property
    def meshio_name(self):
        """This element type's cell-type name in meshio's vocabulary."""
        return "tetra" if self is ElementType.TETRAHEDRON else self.value


class UnstructuredSubMesh(SubMesh):
    """
    Cell-centered finite volume submesh on polygonal/polyhedral elements.

    Supported element types:

    * **2D**: triangles (3 vertices) or quadrilaterals (4 vertices)
    * **3D**: tetrahedra (4 vertices) or hexahedra (8 vertices)

    Hexahedra must have planar faces: volumes and face fluxes are
    ill-defined on warped (non-planar-faced) hexes, so construction
    raises a :class:`pybamm.GeometryError` for them.

    All operators are dimension-agnostic: the same code path handles
    both 2D and 3D, with dimension inferred from ``vertices.shape[1]``.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex coordinates, of shape ``(n_vertices, d)`` (d = 2 or 3).
    elements : numpy.ndarray
        Element vertex indices, of shape ``(n_cells, n_verts_per_cell)``.
        For 2D: 3 (triangles) or 4 (quads).
        For 3D: 4 (tetrahedra) or 8 (hexahedra).
    coord_sys : str, optional
        Coordinate system, default ``"cartesian"``.
    boundary_faces : dict[str, numpy.ndarray] or None, optional
        Maps boundary name to face indices.  If ``None``, no boundary
        tags are assigned: tags must come from the mesh source (the
        built-in generator tags its own box output, file generators use
        the mesh file's boundary names), or call
        :meth:`detect_box_boundaries` for a hand-built axis-aligned box.
    """

    def __init__(self, vertices, elements, coord_sys="cartesian", boundary_faces=None):
        super().__init__()
        self.vertices = np.asarray(vertices, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.dimension = self.vertices.shape[1]
        self.coord_sys = coord_sys

        verts_per_cell = self.elements.shape[1]
        if self.dimension == 2 and verts_per_cell == 4:
            self.element_type = ElementType.QUAD
        elif self.dimension == 2 and verts_per_cell == 3:
            self.element_type = ElementType.TRIANGLE
        elif self.dimension == 3 and verts_per_cell == 4:
            self.element_type = ElementType.TETRAHEDRON
        elif self.dimension == 3 and verts_per_cell == 8:
            self.element_type = ElementType.HEXAHEDRON
        else:
            raise pybamm.GeometryError(
                f"Unsupported: {verts_per_cell} vertices per cell in {self.dimension}D"
            )

        self._compute_cell_geometry()
        self._build_face_connectivity()
        self._compute_face_geometry()

        self.boundary_faces = boundary_faces if boundary_faces is not None else {}

        self.npts = len(self.elements)
        self.npts_lr = self.npts
        self.npts_tb = 1
        self.internal_boundaries = []
        self.interface_data = {}

    # ------------------------------------------------------------------
    # Cell geometry
    # ------------------------------------------------------------------

    def _compute_cell_geometry(self):
        verts = self.vertices[self.elements]  # (n_cells, n_verts, d)
        # Vertex mean is the exact centroid for simplices (triangles, tets);
        # quads and hexes overwrite it with the true area/volume centroid.
        self.cell_centroids = verts.mean(axis=1)

        if self.element_type == ElementType.TRIANGLE:
            v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
            cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (
                v1[:, 1] - v0[:, 1]
            ) * (v2[:, 0] - v0[:, 0])
            self.cell_volumes = 0.5 * np.abs(cross)
        elif self.element_type == ElementType.QUAD:
            # Shoelace formula for arbitrary simple quadrilaterals
            # Vertices ordered: v0, v1, v2, v3 (counterclockwise or clockwise)
            x = verts[:, :, 0]  # (n_cells, 4)
            y = verts[:, :, 1]  # (n_cells, 4)
            # shoelace: sum_i (x_i * y_{i+1} - x_{i+1} * y_i)
            x_next = np.roll(x, -1, axis=1)
            y_next = np.roll(y, -1, axis=1)
            cross = x * y_next - x_next * y
            signed_area = 0.5 * np.sum(cross, axis=1)
            self.cell_volumes = np.abs(signed_area)
            # Polygon centroid (exact for non-parallelogram quads, where the
            # vertex mean is not)
            cx = np.sum((x + x_next) * cross, axis=1) / (6.0 * signed_area)
            cy = np.sum((y + y_next) * cross, axis=1) / (6.0 * signed_area)
            self.cell_centroids = np.column_stack([cx, cy])
        elif self.element_type == ElementType.TETRAHEDRON:
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
        elif self.element_type == ElementType.HEXAHEDRON:
            # 5-tet decomposition, exact only for planar-faced hexes (warped
            # faces are rejected in _compute_face_geometry). Tet volumes are
            # summed signed so a degenerate or inverted cell cannot cancel
            # into a plausible positive volume.
            t = verts[:, self._HEX_TETS]  # (n_cells, 5, 4, d)
            d1 = t[:, :, 1] - t[:, :, 0]
            d2 = t[:, :, 2] - t[:, :, 0]
            d3 = t[:, :, 3] - t[:, :, 0]
            tet_vols = np.einsum("ijk,ijk->ij", d1, np.cross(d2, d3)) / 6.0
            vol = tet_vols.sum(axis=1)
            self.cell_volumes = np.abs(vol)
            # Volume-weighted tet centroids: exact for planar-faced hexes
            # such as frusta, where the vertex mean is not
            moment = np.einsum("ij,ijk->ik", tet_vols, t.mean(axis=2))
            safe_vol = np.where(vol == 0.0, 1.0, vol)
            nonzero = (vol != 0.0)[:, None]
            self.cell_centroids = np.where(
                nonzero, moment / safe_vol[:, None], self.cell_centroids
            )

    # ------------------------------------------------------------------
    # Face-cell connectivity
    # ------------------------------------------------------------------

    def _build_face_connectivity(self):
        """Extract faces, identify internal / boundary, record owner-neighbor."""
        if self.element_type == ElementType.HEXAHEDRON:
            n_verts_per_face = 4
        else:
            n_verts_per_face = self.dimension

        elems = self.elements
        n_cells = len(elems)

        # Build all faces at once using local face definitions
        if self.element_type == ElementType.QUAD:
            idx = np.arange(4)
            local = np.stack([idx, (idx + 1) % 4], axis=1)  # (4, 2)
        elif self.element_type == ElementType.TRIANGLE:
            local = np.array([[1, 2], [0, 2], [0, 1]])  # skip vertex 0, 1, 2
        elif self.element_type == ElementType.TETRAHEDRON:
            local = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        elif self.element_type == ElementType.HEXAHEDRON:
            local = np.array(self._HEX_FACES)
        n_fpc = len(local)

        all_faces = elems[:, local].reshape(-1, n_verts_per_face)
        cell_ids = np.repeat(np.arange(n_cells), n_fpc)

        # Canonical keys: sort vertex indices within each face
        sorted_faces = np.sort(all_faces, axis=1)

        # Find unique faces and which are shared (internal) vs single (boundary)
        _, inverse, counts = np.unique(
            sorted_faces, axis=0, return_inverse=True, return_counts=True
        )
        # numpy 2.0.0 (the declared floor) returns inverse with shape (n, 1);
        # 2.0.1+ returns (n,). Flatten so masks below stay 1-D everywhere.
        inverse = inverse.reshape(-1)

        # A manifold mesh shares each face between at most two cells. A count of
        # three or more means overlapping or non-conforming elements; such faces
        # match neither branch below and would silently vanish, so reject them.
        if (counts > 2).any():
            raise pybamm.GeometryError(
                "Unstructured mesh is non-manifold: at least one face is shared "
                "by more than two cells. Check for overlapping or duplicated "
                "elements in the input mesh."
            )

        is_internal = counts[inverse] == 2
        is_boundary = counts[inverse] == 1

        # For internal faces, we need owner/neighbor pairs.
        # Group by unique face index; first occurrence is owner, second is neighbor.
        internal_mask = is_internal
        int_inv = inverse[internal_mask]
        int_cells = cell_ids[internal_mask]
        int_faces_raw = all_faces[internal_mask]

        # Sort by unique-face-id to pair them up: [owner0, neighbor0, owner1, neighbor1, ...]
        order = np.argsort(int_inv, kind="stable")
        int_cells_sorted = int_cells[order]
        int_faces_sorted = int_faces_raw[order]

        internal_owner = int_cells_sorted[0::2]
        internal_neighbor = int_cells_sorted[1::2]
        internal_face_verts = int_faces_sorted[0::2]

        # Boundary faces
        bnd_face_verts = all_faces[is_boundary]
        bnd_owners = cell_ids[is_boundary]

        n_internal = len(internal_owner)
        n_boundary = len(bnd_owners)

        self.faces = np.concatenate([internal_face_verts, bnd_face_verts], axis=0)
        self.face_owner = np.concatenate([internal_owner, bnd_owners])
        self.face_neighbor = internal_neighbor
        self.n_internal_faces = n_internal
        self._n_boundary_faces = n_boundary
        self._boundary_face_start = n_internal

    # Standard hex vertex ordering:
    #   0=(i,j,k)   1=(i+1,j,k)   2=(i+1,j+1,k)   3=(i,j+1,k)
    #   4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i+1,j+1,k+1) 7=(i,j+1,k+1)
    _HEX_FACES = [
        (0, 3, 7, 4),  # x- (left)
        (1, 2, 6, 5),  # x+ (right)
        (0, 1, 5, 4),  # y- (front)
        (3, 2, 6, 7),  # y+ (back)
        (0, 1, 2, 3),  # z- (bottom)
        (4, 5, 6, 7),  # z+ (top)
    ]

    # 5-tet split for volume/centroid computation, each tet ordered so its
    # signed volume is positive for a right-handed hex
    _HEX_TETS = np.array(
        [(0, 1, 2, 5), (0, 2, 3, 7), (0, 5, 7, 4), (2, 7, 5, 6), (0, 5, 2, 7)]
    )

    # ------------------------------------------------------------------
    # Face geometry
    # ------------------------------------------------------------------

    def _compute_face_geometry(self):
        face_verts = self.vertices[self.faces]

        self.face_centroids = face_verts.mean(axis=1)

        if self.dimension == 2:
            v0, v1 = face_verts[:, 0], face_verts[:, 1]
            edge = v1 - v0
            self.face_areas = np.linalg.norm(edge, axis=1)
            normals = np.column_stack([edge[:, 1], -edge[:, 0]])
        elif self.element_type == ElementType.HEXAHEDRON:
            # Face = quad: 4 vertices. Area via cross product of diagonals.
            v0 = face_verts[:, 0]
            v1 = face_verts[:, 1]
            v2 = face_verts[:, 2]
            v3 = face_verts[:, 3]

            # Quad areas, normals, and the 5-tet cell volumes are only
            # well-defined when all four vertices of a face are coplanar,
            # so warped (non-planar) faces are rejected outright.
            plane_normal = np.cross(v1 - v0, v2 - v0)
            plane_norm = np.linalg.norm(plane_normal, axis=1)
            safe_norm = np.where(plane_norm < 1e-30, 1.0, plane_norm)
            offset = np.abs(
                np.einsum("ij,ij->i", v3 - v0, plane_normal / safe_norm[:, None])
            )
            diag_len = np.linalg.norm(v2 - v0, axis=1)
            warped = offset > 1e-8 * np.maximum(diag_len, 1e-30)
            if warped.any():
                raise pybamm.GeometryError(
                    f"{int(warped.sum())} hexahedral face(s) are non-planar "
                    "(warped): the fourth vertex does not lie in the plane of "
                    "the other three. Volumes and face fluxes are ill-defined "
                    "on warped hexahedra. Fix the mesh, or use tetrahedra."
                )

            diag1 = v2 - v0
            diag2 = v3 - v1
            cross = np.cross(diag1, diag2)
            self.face_areas = 0.5 * np.linalg.norm(cross, axis=1)
            normals = cross

            # Area-weighted centroid over the (0,1,2)/(0,2,3) triangle split:
            # exact for planar non-parallelogram quads (e.g. trapezoids),
            # where the vertex mean is not. Signed areas taken along the face
            # normal keep non-convex planar quads exact too.
            n_hat = plane_normal / safe_norm[:, None]
            area1 = 0.5 * np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), n_hat)
            area2 = 0.5 * np.einsum("ij,ij->i", np.cross(v2 - v0, v3 - v0), n_hat)
            total = area1 + area2
            safe_total = np.where(np.abs(total) < 1e-30, 1.0, total)
            weighted = (
                area1[:, None] * (v0 + v1 + v2) + area2[:, None] * (v0 + v2 + v3)
            ) / (3.0 * safe_total[:, None])
            nonzero = np.abs(total) >= 1e-30
            self.face_centroids[nonzero] = weighted[nonzero]
        else:
            # Face = triangle: 3 vertices
            v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
            cross = np.cross(v1 - v0, v2 - v0)
            self.face_areas = 0.5 * np.linalg.norm(cross, axis=1)
            normals = cross

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

    def detect_box_boundaries(self):
        """Tag boundary faces of an axis-aligned box by outward normal.

        Assigns ``left``/``right`` (x), ``front``/``back`` (y, 3D only),
        and ``bottom``/``top`` (z) by each exterior face's dominant normal
        direction, overwriting ``boundary_faces``. Only meaningful for
        axis-aligned box domains (each face genuinely normal to one axis);
        on curved geometry the buckets are not surfaces. The built-in
        generator calls this on its output; meshes from files should carry
        their own boundary names instead.
        """
        bnd_start = self._boundary_face_start
        bnd_centroids = self.face_centroids[bnd_start:]

        if len(bnd_centroids) == 0:
            self.boundary_faces = {}
            return

        # Classify every external face by its outward normal direction so all
        # protrusions (e.g., tabs) get assigned a BC bucket.
        bnd_normals = self.face_normals[bnd_start:]
        n_bnd = len(bnd_normals)

        # A zero-area face has no normal direction and would land in an
        # arbitrary bucket; surface it instead.
        if np.any(self.face_areas[bnd_start:] < 1e-30):
            raise pybamm.GeometryError(
                "Mesh has degenerate (zero-area) boundary faces; boundary "
                "detection cannot classify them."
            )

        axis = np.argmax(np.abs(bnd_normals), axis=1)
        positive = (bnd_normals[np.arange(n_bnd), axis] >= 0).astype(int)
        if self.dimension == 3:
            names = ["left", "right", "front", "back", "bottom", "top"]
            slot = axis * 2 + positive
        else:
            names = ["left", "right", "bottom", "top"]
            slot = np.where(axis == 0, positive, 2 + positive)

        self.boundary_faces = {}
        for k, name in enumerate(names):
            indices = np.nonzero(slot == k)[0] + bnd_start
            if len(indices) > 0:
                self.boundary_faces[name] = indices

    # ------------------------------------------------------------------
    # Combining domains
    # ------------------------------------------------------------------

    @classmethod
    def combine(cls, submeshes):
        """
        Weld a list of :class:`UnstructuredSubMesh` objects into one mesh.

        Coincident boundary nodes at domain interfaces are merged so that
        face connectivity spans across domains.

        Parameters
        ----------
        submeshes : list of UnstructuredSubMesh
            The submeshes to combine, in order.

        Returns
        -------
        UnstructuredSubMesh
            A single mesh covering the union of the input domains.
        """
        from scipy.spatial import cKDTree

        element_types = {sm.element_type for sm in submeshes}
        if len(element_types) > 1:
            raise pybamm.GeometryError(
                f"Cannot combine unstructured submeshes of different element "
                f"types: {sorted(t.value for t in element_types)}. All domains "
                f"must use the same element type."
            )

        # Weld coincident nodes across submeshes regardless of which face tag
        # they belong to, so that interfaces of arbitrary topology (star, tree,
        # graph) become internal faces and TPFA handles cross-region flux
        # without internal Neumann book-keeping.
        tol = _geometric_tolerance(submeshes)
        combined_nodes = np.asarray(submeshes[0].vertices, dtype=float)
        global_maps = [np.arange(len(combined_nodes))]

        for sm in submeshes[1:]:
            d, j = cKDTree(combined_nodes).query(sm.vertices)
            new = d >= tol
            global_maps.append(
                np.where(new, np.cumsum(new) - 1 + len(combined_nodes), j)
            )
            combined_nodes = np.vstack([combined_nodes, sm.vertices[new]])

        combined_elements = np.concatenate(
            [gm[sm.elements] for gm, sm in zip(global_maps, submeshes, strict=True)],
            axis=0,
        )
        combined = cls(
            combined_nodes,
            combined_elements,
            coord_sys=submeshes[0].coord_sys,
        )

        # The combined mesh starts with no tags: every tag is propagated from
        # the input submeshes by matching boundary face centroids. Faces that
        # welding turned internal (interface faces) match no combined boundary
        # face and drop out naturally.
        tag_centroids = {}  # tag -> list of centroid arrays
        for sm in submeshes:
            for tag, face_indices in sm.boundary_faces.items():
                tag_centroids.setdefault(tag, []).append(
                    sm.face_centroids[face_indices]
                )

        if tag_centroids:
            bnd_start = combined._boundary_face_start
            bnd_centroids = combined.face_centroids[bnd_start:]
            if len(bnd_centroids) > 0:
                tree = cKDTree(bnd_centroids)
                # Welding may move nodes (and so centroids) by up to the weld
                # tolerance, so tag matching must never be tighter than it
                match_tol = tol
                for tag, centroid_list in tag_centroids.items():
                    all_src = np.concatenate(centroid_list, axis=0)
                    dists, idxs = tree.query(all_src)
                    matched = idxs[dists < match_tol]
                    if len(matched) > 0:
                        combined.boundary_faces[tag] = np.unique(matched) + bnd_start

        # Welding only produces internal faces where interface nodes coincide.
        # If a region shares no interface with the rest (mismatched transverse
        # grids, wrong units, or a non-conforming input mesh) it stays a
        # separate connected component and no flux can cross into it, which
        # otherwise solves silently to a wrong answer. Reuse the integer face
        # connectivity (no distance tolerance) to require a single component.
        if len(submeshes) > 1 and combined.npts > 1:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components

            n_int = combined._boundary_face_start
            rows = np.concatenate([combined.face_owner[:n_int], combined.face_neighbor])
            cols = np.concatenate([combined.face_neighbor, combined.face_owner[:n_int]])
            adjacency = csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(combined.npts, combined.npts),
            )
            n_components, _ = connected_components(adjacency, directed=False)
            if n_components > 1:
                raise pybamm.GeometryError(
                    f"Combined unstructured mesh has {n_components} disconnected "
                    f"regions: welding produced no interface between some domains, "
                    f"so no flux could cross and the solve would be silently wrong. "
                    f"Adjacent domains must share a conforming interface — matching "
                    f"transverse grids and coordinate units, or a fragmented "
                    f"(node-shared) mesh from the mesh generator."
                )

        return combined

    def optimize_ordering(self):
        """Reorder cells using Reverse Cuthill-McKee to reduce Jacobian bandwidth.

        Permutes all cell-indexed arrays (elements, centroids, volumes,
        face_owner, face_neighbor, interface_data) so that adjacent cells
        have nearby indices, minimising the bandwidth of the FVM
        connectivity matrix.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        n = self.npts
        if n <= 1:
            return

        n_int = self._boundary_face_start
        owners = self.face_owner[:n_int]
        neighbors = self.face_neighbor

        rows = np.concatenate([owners, neighbors])
        cols = np.concatenate([neighbors, owners])
        data = np.ones(len(rows), dtype=np.float64)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        perm = reverse_cuthill_mckee(adj)
        inv_perm = np.empty(n, dtype=int)
        inv_perm[perm] = np.arange(n)

        self.elements = self.elements[perm]
        self.cell_centroids = self.cell_centroids[perm]
        self.cell_volumes = self.cell_volumes[perm]

        self.face_owner = inv_perm[self.face_owner]
        self.face_neighbor = inv_perm[self.face_neighbor]

        # In interface_data, "left_cells" indexes this mesh and "right_cells"
        # indexes the neighbour, so only the former is permuted. The neighbour
        # holds a mirrored view of the same pairing and is updated with it.
        for data_dict in self.interface_data.values():
            permuted = inv_perm[data_dict["left_cells"]]
            data_dict["left_cells"] = permuted
            other = data_dict.get("other_mesh")
            if other is None:
                continue
            for mirror in other.interface_data.values():
                if mirror.get("other_mesh") is self:
                    mirror["right_cells"] = permuted

    def boundary_loops(self):
        """Return boundary loops as a list of ``matplotlib.path.Path`` (2D only).

        Walks boundary edges to extract one or more closed loops.  The first
        path is the outer boundary (largest area); subsequent paths are holes.
        Use this to test containment: a point is in the domain if it is inside
        the outer loop and outside all hole loops.
        """
        if self.dimension != 2:
            return None

        from matplotlib.path import Path

        bnd_start = self._boundary_face_start
        bnd_edges = self.faces[bnd_start:]
        if len(bnd_edges) == 0:
            return None

        adj: dict[int, list[tuple[int, int]]] = {}
        for i, edge in enumerate(bnd_edges):
            v0, v1 = int(edge[0]), int(edge[1])
            adj.setdefault(v0, []).append((i, v1))
            adj.setdefault(v1, []).append((i, v0))

        visited: set[int] = set()
        loops: list[list[int]] = []

        for start_edge_idx in range(len(bnd_edges)):
            if start_edge_idx in visited:
                continue
            start_v = int(bnd_edges[start_edge_idx][0])
            loop = [start_v]
            current = start_v
            while True:
                found = False
                for edge_idx, next_v in adj[current]:
                    if edge_idx not in visited:
                        visited.add(edge_idx)
                        loop.append(next_v)
                        current = next_v
                        found = True
                        break
                if not found:
                    break
            loops.append(loop)

        def signed_area(pts):
            x, y = pts[:, 0], pts[:, 1]
            return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

        loop_data = []
        for loop in loops:
            pts = self.vertices[loop]
            sa = signed_area(pts)
            loop_data.append((abs(sa), pts))

        loop_data.sort(key=lambda t: t[0], reverse=True)

        paths = []
        for pts in (ld[1] for ld in loop_data):
            codes = [Path.LINETO] * len(pts)
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            paths.append(Path(pts, codes))
        return paths

    def contains_points_3d(self, query_pts):
        """Test whether 3D points lie inside the mesh domain.

        Uses the generalized winding number (Van Oosterom--Strackee signed
        solid angle sum over all boundary triangles).  Points inside the
        domain return ``True``; points outside or inside internal cavities
        return ``False``.
        """
        query_pts = np.asarray(query_pts, dtype=np.float64)
        bnd_start = self._boundary_face_start
        bnd_fv = self.faces[bnd_start:]
        bnd_normals = self.face_normals[bnd_start:]
        n_vpf = bnd_fv.shape[1]

        if n_vpf == 3:
            tri_idx = bnd_fv
            tri_normals = bnd_normals
        elif n_vpf == 4:
            tri_idx = np.concatenate(
                [bnd_fv[:, [0, 1, 2]], bnd_fv[:, [0, 2, 3]]], axis=0
            )
            tri_normals = np.concatenate([bnd_normals, bnd_normals], axis=0)
        else:
            raise pybamm.GeometryError(
                f"contains_points_3d: unsupported face with {n_vpf} vertices"
            )

        v0 = self.vertices[tri_idx[:, 0]]
        v1 = self.vertices[tri_idx[:, 1]]
        v2 = self.vertices[tri_idx[:, 2]]

        # Ensure consistent CCW orientation from outside (matching outward normals)
        cross = np.cross(v1 - v0, v2 - v0)
        flip = np.sum(cross * tri_normals, axis=1) < 0
        v1_fixed = v1.copy()
        v2_fixed = v2.copy()
        v1_fixed[flip] = v2[flip]
        v2_fixed[flip] = v1[flip]

        n_query = len(query_pts)
        winding = np.zeros(n_query)

        # Loop over query points (usually few), vectorised over the many
        # boundary triangles — the reverse nesting costs O(n_triangles)
        # per point regardless of how few points are asked about.
        for i in range(n_query):
            a = v0 - query_pts[i]
            b = v1_fixed - query_pts[i]
            c = v2_fixed - query_pts[i]

            an = np.linalg.norm(a, axis=1)
            bn = np.linalg.norm(b, axis=1)
            cn = np.linalg.norm(c, axis=1)

            num = np.einsum("ij,ij->i", a, np.cross(b, c))
            den = (
                an * bn * cn
                + np.einsum("ij,ij->i", a, b) * cn
                + np.einsum("ij,ij->i", a, c) * bn
                + np.einsum("ij,ij->i", b, c) * an
            )

            winding[i] = 2.0 * np.arctan2(num, den).sum()

        return winding > 2.0 * np.pi


# ======================================================================
# Mesh generators
# ======================================================================


class UnstructuredMeshGenerator(MeshGenerator):
    """
    Built-in generator that creates meshes from structured grids.

    * **2D**: rectangular domain meshed as quads, or triangulated by
      splitting each quad into 2 triangles.
    * **3D**: rectangular prism meshed as hexahedra, or split into 6
      tetrahedra per hex (Kuhn decomposition).

    On box domains, hexahedra (the 3D default) are preferred for real
    simulations: same resolution with 6x fewer cells and ideal face
    orthogonality. The simplex element types chiefly exercise the same
    code paths as user-supplied (e.g. gmsh) meshes without needing a
    mesh file, which is useful for testing and validation.

    Parameters
    ----------
    coord_sys : str, optional
        Coordinate system, default ``"cartesian"``.
    element_type : str, optional
        ``"quad"`` or ``"triangle"`` in 2D; ``"hexahedron"`` or
        ``"tetrahedron"`` in 3D. If ``None``, defaults to ``"triangle"``
        in 2D and ``"hexahedron"`` in 3D.
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
            raise pybamm.GeometryError(
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

        etype = self._element_type or ElementType.TRIANGLE
        if etype == ElementType.QUAD:
            nodes, elements = _make_quad_grid(x_edges, z_edges)
        elif etype == ElementType.TRIANGLE:
            nodes, elements = _quad_to_tri(x_edges, z_edges)
        else:
            raise pybamm.GeometryError(f"Unsupported 2D element_type: {etype!r}")
        submesh = UnstructuredSubMesh(nodes, elements, coord_sys=self.coord_sys)
        # The generator's output is an axis-aligned box by construction
        submesh.detect_box_boundaries()
        return submesh

    # ------------------------------------------------------------------
    # 3D
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

        etype = self._element_type or ElementType.HEXAHEDRON
        if etype == ElementType.HEXAHEDRON:
            nodes, elements = _hex_grid(x_edges, y_edges, z_edges)
        elif etype == ElementType.TETRAHEDRON:
            nodes, elements = _hex_to_tet(x_edges, y_edges, z_edges)
        else:
            raise pybamm.GeometryError(f"Unsupported 3D element_type: {etype!r}")
        submesh = UnstructuredSubMesh(nodes, elements, coord_sys=self.coord_sys)
        # The generator's output is an axis-aligned box by construction
        submesh.detect_box_boundaries()
        return submesh


class UserSuppliedUnstructuredMesh(MeshGenerator):
    """
    Load an unstructured mesh from an external file via *meshio*.

    Supported cell types are tetrahedra (3D) and triangles or quadrilaterals
    (2D). Hexahedral file meshes are rejected: file meshes commonly contain
    warped (non-planar-faced) hexes, whose volumes and face fluxes are
    ill-defined. Convert such meshes to tetrahedra before loading.

    The interface between adjacent domains must be **conforming**: the two
    sides must share the same interface nodes, so that welding in
    :meth:`UnstructuredSubMesh.combine` turns the interface into internal
    faces. In gmsh, build the regions from one geometry or fragment the parts
    (``BooleanFragments`` / ``Coherence``) so the shared surface is meshed
    once. A non-conforming interface raises a :class:`pybamm.GeometryError`
    when the domains are combined.

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
    merge_tolerance : float or None, optional
        Absolute length (in the mesh file's units) below which coincident
        nodes across cell blocks are welded, by quantising coordinates to
        a grid of this spacing. Default ``1e-12``; pass ``None`` or ``0``
        to disable welding.
    """

    def __init__(
        self,
        filepath,
        subdomain_mapping=None,
        boundary_mapping=None,
        coord_sys="cartesian",
        merge_tolerance=1e-12,
    ):
        self.submesh_type = UnstructuredSubMesh
        self.submesh_params = {}
        self.filepath = filepath
        self.subdomain_mapping = subdomain_mapping or {}
        self.boundary_mapping = boundary_mapping or {}
        self.coord_sys = coord_sys
        self.merge_tolerance = merge_tolerance
        self._cached_mesh = None

    def __call__(self, lims, npts):
        meshio = pybamm.import_optional_dependency("meshio")

        if self._cached_mesh is None:
            self._cached_mesh = meshio.read(self.filepath)

        mesh = self._cached_mesh
        nodes = mesh.points

        # Determine which domain is being requested from the lims keys
        domain_name = self._domain_name_from_lims(lims)

        # Extract supported cells (triangles/quads or tets/hexes)
        cells, cell_type = self._extract_supported_cells(mesh)

        if domain_name and domain_name in self.subdomain_mapping:
            tag_value = self.subdomain_mapping[domain_name]
            cell_mask = self._get_cell_mask(mesh, cell_type, tag_value)
            elements = cells[cell_mask]
        else:
            elements = cells

        # Weld coincident nodes across cell blocks so touching regions
        # (e.g. body-tab interfaces) are thermally connected.
        if self.merge_tolerance is not None and self.merge_tolerance > 0:
            scale = 1.0 / self.merge_tolerance
            quantized = np.round(nodes * scale).astype(np.int64)
            _, unique_idx, inverse = np.unique(
                quantized, axis=0, return_index=True, return_inverse=True
            )
            # numpy 2.0.0 returns inverse with shape (n, 1); 2.0.1+ (n,)
            inverse = inverse.reshape(-1)
            nodes = nodes[unique_idx]
            elements = inverse[elements]

        # Re-index nodes to compact numbering
        unique_nodes = np.unique(elements)
        node_map = np.full(nodes.shape[0], -1, dtype=int)
        node_map[unique_nodes] = np.arange(len(unique_nodes))
        compact_nodes = nodes[unique_nodes]
        compact_elements = node_map[elements]

        # Trim to 2D if all z-coordinates are zero
        if compact_nodes.shape[1] == 3 and np.allclose(compact_nodes[:, 2], 0):
            compact_nodes = compact_nodes[:, :2]

        submesh = UnstructuredSubMesh(
            compact_nodes, compact_elements, coord_sys=self.coord_sys
        )

        if self.boundary_mapping:
            facet_type = "triangle" if cell_type == ElementType.TETRAHEDRON else "line"
            facets, facet_tags = _extract_tagged_facets(mesh, facet_type)
            if facets is None:
                pybamm.logger.warning(
                    f"boundary_mapping given but no tagged '{facet_type}' "
                    f"facets found in {self.filepath}; no boundary tags set"
                )
            else:
                if self.merge_tolerance is not None and self.merge_tolerance > 0:
                    facets = inverse[facets]
                facets = node_map[facets]
                # Facets of other subdomains reference nodes outside this
                # submesh and cannot match
                in_domain = (facets >= 0).all(axis=1)
                for name, tag in self.boundary_mapping.items():
                    matched = _match_facets_to_boundary_faces(
                        facets[in_domain & (facet_tags == tag)], submesh
                    )
                    if len(matched) > 0:
                        submesh.boundary_faces[name] = matched
                    else:
                        pybamm.logger.warning(
                            f"boundary_mapping entry {name!r} (tag {tag}) "
                            f"matched no boundary faces of this submesh"
                        )

        return submesh

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
    def _extract_supported_cells(mesh):
        # Hexahedra from files are rejected outright rather than silently
        # dropped: file meshes commonly contain warped (non-planar-faced)
        # hexes, for which cell volumes and face fluxes are ill-defined.
        if any(
            block.type == ElementType.HEXAHEDRON.meshio_name for block in mesh.cells
        ):
            raise pybamm.GeometryError(
                "Hexahedral cells in mesh files are not supported: warped "
                "(non-planar-faced) hexahedra have ill-defined volumes and "
                "face fluxes. Convert the mesh to tetrahedra (e.g. with "
                "gmsh or meshio) and reload. Hexahedral meshes are still "
                "available through pybamm.UnstructuredMeshGenerator, whose "
                "axis-aligned cells are always well-defined."
            )
        # Prefer 3D cells when present, otherwise fall back to 2D.
        for cell_type in (
            ElementType.TETRAHEDRON,
            ElementType.TRIANGLE,
            ElementType.QUAD,
        ):
            blocks = [
                block.data
                for block in mesh.cells
                if block.type == cell_type.meshio_name
            ]
            if blocks:
                if len(blocks) == 1:
                    return blocks[0], cell_type
                return np.concatenate(blocks, axis=0), cell_type
        raise pybamm.GeometryError(
            "No supported cells found in mesh file (expected tetra/triangle/quad)"
        )

    @staticmethod
    def _get_cell_mask(mesh, cell_type, tag_value):
        for data_list in mesh.cell_data.values():
            matching = [
                data
                for block, data in zip(mesh.cells, data_list, strict=False)
                if block.type == cell_type.meshio_name
            ]
            if matching:
                if len(matching) == 1:
                    return matching[0] == tag_value
                return np.concatenate(matching, axis=0) == tag_value
        raise pybamm.GeometryError(
            f"Could not find cell data tag {tag_value} for cell type '{cell_type}'"
        )


# ======================================================================
# Tagged-region mesh generator
# ======================================================================


class TaggedSubMeshGenerator(MeshGenerator):
    """
    Build an :class:`UnstructuredSubMesh` from cells of a single Gmsh
    physical group in a ``.msh`` file.

    Use one instance per region in a multi-domain pybamm model — the
    region name doubles as the pybamm domain name. Compare to
    :class:`UserSuppliedUnstructuredMesh`, which routes multiple regions
    through one generator by introspecting ``lims``; ``TaggedSubMeshGenerator``
    is simpler when the model already supplies one mesh generator per
    domain.

    Regions that share an interface must be conforming across it (the shared
    surface meshed once, so both regions reference the same interface nodes),
    or combining the domains raises a :class:`pybamm.GeometryError`. Fragment
    the geometry in gmsh (``BooleanFragments`` / ``Coherence``) to guarantee
    this.

    Parameters
    ----------
    region : str
        Gmsh physical-group name (key in ``meshio.read(...).field_data``).
    mesh_path : str or pathlib.Path
        Path to the ``.msh`` file.
    scale : float, optional
        Multiplier applied to mesh node coordinates (e.g. ``1e-3`` to
        convert mm to m). Default ``1.0``.
    coord_sys : str, optional
        Coordinate system label, default ``"cartesian"``.
    boundary_mapping : dict[str, str or int] or None, optional
        Maps boundary name to a gmsh physical *surface* group, given as
        its ``field_data`` name or integer tag. Matching tagged surface
        triangles become the named entries in ``boundary_faces``. Without
        it the submesh carries no boundary tags.
    """

    _mesh_cache: dict = {}

    def __init__(
        self, region, mesh_path, scale=1.0, coord_sys="cartesian", boundary_mapping=None
    ):
        self.submesh_type = UnstructuredSubMesh
        self.submesh_params = {}
        self._mesh_path = mesh_path
        self._region = region
        self._scale = float(scale)
        self.coord_sys = coord_sys
        self.boundary_mapping = boundary_mapping or {}

    @classmethod
    def _read(cls, path):
        if path not in cls._mesh_cache:
            meshio = pybamm.import_optional_dependency("meshio")
            cls._mesh_cache[path] = meshio.read(str(path))
        return cls._mesh_cache[path]

    def __call__(self, lims, npts):
        m = self._read(self._mesh_path)
        if self._region not in m.field_data:
            raise pybamm.GeometryError(
                f"region {self._region!r} not in mesh field_data; "
                f"available: {list(m.field_data)}"
            )
        tag_id = int(m.field_data[self._region][0])

        tet_blocks = []
        for block, tags in zip(
            m.cells, m.cell_data.get("gmsh:physical", []), strict=False
        ):
            if block.type != ElementType.TETRAHEDRON.meshio_name:
                continue
            mask = np.asarray(tags, dtype=np.int32) == tag_id
            if mask.any():
                tet_blocks.append(block.data[mask])
        if not tet_blocks:
            raise pybamm.GeometryError(f"no tets for region {self._region!r}")

        elements = np.concatenate(tet_blocks, axis=0)
        unique_nodes = np.unique(elements)
        node_map = np.full(m.points.shape[0], -1, dtype=np.int64)
        node_map[unique_nodes] = np.arange(len(unique_nodes))
        nodes = m.points[unique_nodes] * self._scale
        submesh = UnstructuredSubMesh(
            nodes, node_map[elements], coord_sys=self.coord_sys
        )

        if self.boundary_mapping:
            facets, facet_tags = _extract_tagged_facets(m, "triangle")
            if facets is None:
                pybamm.logger.warning(
                    f"boundary_mapping given but no tagged surface triangles "
                    f"found in {self._mesh_path}; no boundary tags set"
                )
            else:
                facets = node_map[facets]
                in_domain = (facets >= 0).all(axis=1)
                for name, group in self.boundary_mapping.items():
                    if isinstance(group, str):
                        if group not in m.field_data:
                            raise pybamm.GeometryError(
                                f"boundary group {group!r} not in mesh "
                                f"field_data; available: {list(m.field_data)}"
                            )
                        group = int(m.field_data[group][0])
                    matched = _match_facets_to_boundary_faces(
                        facets[in_domain & (facet_tags == group)], submesh
                    )
                    if len(matched) > 0:
                        submesh.boundary_faces[name] = matched
                    else:
                        pybamm.logger.warning(
                            f"boundary_mapping entry {name!r} matched no "
                            f"boundary faces of region {self._region!r}"
                        )

        return submesh


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

    Domains are assumed to be stacked along x: faces are paired by their
    transverse (non-x) centroid coordinates.

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
        raise pybamm.GeometryError(
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

    tol = _geometric_tolerance([left_mesh, right_mesh])
    if np.any(dists > tol):
        raise pybamm.GeometryError(
            f"Interface faces do not match: max transverse mismatch = {dists.max():.2e}. "
            "Ensure both meshes have the same transverse grid."
        )

    # The nearest-neighbour query is one-directional: without this check a
    # surplus right face silently loses its flux, and a doubly-claimed one
    # double-counts it.
    if len(left_bnd) != len(right_bnd) or len(np.unique(right_indices)) != len(
        right_indices
    ):
        raise pybamm.GeometryError(
            f"Interface face pairing is not one-to-one: {len(left_bnd)} 'right' "
            f"faces on the left mesh vs {len(right_bnd)} 'left' faces on the "
            f"right mesh. Both sides must expose the same interface faces."
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
        "other_mesh": right_mesh,
    }

    if right_name is not None:
        left_mesh.interface_data[right_name] = result
    if left_name is not None:
        right_mesh.interface_data[left_name] = {
            "left_cells": right_cells,
            "right_cells": left_cells,
            "face_areas": face_areas,
            "cell_distances": cell_distances,
            "other_mesh": left_mesh,
        }

    return result


# ======================================================================
# Geometric tolerance
# ======================================================================


def _geometric_tolerance(submeshes, rel=1e-3):
    """Distance below which two points of these meshes are the same point.

    Scaled to the smallest sampled element edge: distinct nodes (and face
    centroids) are at least one edge length apart, so a small fraction of
    it can never merge genuinely distinct entities, while absorbing the
    interface jitter that reduced-precision mesh files and unit
    conversions produce. An absolute tolerance cannot do both across mesh
    scales — battery meshes in SI units are ~1e-4 m across.
    """
    min_edge = min(
        float(
            np.linalg.norm(
                sm.vertices[sm.elements[:, 1]] - sm.vertices[sm.elements[:, 0]], axis=1
            ).min()
        )
        for sm in submeshes
    )
    return rel * min_edge


# ======================================================================
# Boundary facet tagging helpers
# ======================================================================


def _extract_tagged_facets(mesh, facet_type):
    """Concatenate a meshio mesh's facet blocks and their integer tags.

    Returns ``(facets, tags)`` arrays, or ``(None, None)`` if the mesh has
    no facet blocks of ``facet_type`` or no cell data to tag them with.
    Prefers the ``gmsh:physical`` cell-data key, falling back to the first
    available key.
    """
    block_ids = [i for i, b in enumerate(mesh.cells) if b.type == facet_type]
    if not block_ids:
        return None, None
    data_lists = mesh.cell_data.get("gmsh:physical")
    if data_lists is None:
        data_lists = next(iter(mesh.cell_data.values()), None)
    if data_lists is None:
        return None, None
    facets = np.concatenate([mesh.cells[i].data for i in block_ids], axis=0)
    tags = np.concatenate([np.asarray(data_lists[i]) for i in block_ids])
    return facets, tags


def _match_facets_to_boundary_faces(facets, submesh):
    """Return submesh boundary-face indices whose vertex sets match ``facets``."""
    bnd_start = submesh._boundary_face_start
    lookup = {
        tuple(face): bnd_start + i
        for i, face in enumerate(np.sort(submesh.faces[bnd_start:], axis=1).tolist())
    }
    matched = {
        lookup[key]
        for key in map(tuple, np.sort(facets, axis=1).tolist())
        if key in lookup
    }
    return np.array(sorted(matched), dtype=int)


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

    i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    n0 = (i * (nz + 1) + j).ravel()
    elements = np.column_stack([n0, n0 + (nz + 1), n0 + (nz + 2), n0 + 1])

    return nodes, elements


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

    i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    n0 = (i * (nz + 1) + j).ravel()
    elements = np.empty((2 * len(n0), 3), dtype=int)
    elements[0::2] = np.column_stack([n0, n0 + (nz + 1), n0 + (nz + 2)])
    elements[1::2] = np.column_stack([n0, n0 + (nz + 2), n0 + 1])

    return nodes, elements


def _hex_grid(x_edges, y_edges, z_edges):
    """
    Create a hexahedral grid from edge arrays.

    Returns nodes and 8-vertex hex elements suitable for
    :class:`UnstructuredSubMesh` with ``element_type="hexahedron"``.

    Vertex ordering per hex matches :attr:`UnstructuredSubMesh._HEX_FACES`:

    ::

        0=(i,j,k)   1=(i+1,j,k)   2=(i+1,j+1,k)   3=(i,j+1,k)
        4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i+1,j+1,k+1) 7=(i,j+1,k+1)

    Returns
    -------
    nodes : (n_nodes, 3)
    elements : (n_cells, 8)
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1

    xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Loop order determines cell numbering and hence Jacobian bandwidth.
    # Bandwidth = product of the two fastest-varying dimension sizes.
    # Minimise by putting the largest dimension outermost (slowest).
    dims = sorted([(nx, "x"), (ny, "y"), (nz, "z")], key=lambda d: d[0], reverse=True)
    grids = np.meshgrid(*(np.arange(n) for n, _ in dims), indexing="ij")
    idx = {name: grid.ravel() for (_, name), grid in zip(dims, grids, strict=True)}
    i, j, k = idx["x"], idx["y"], idx["z"]

    n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    p = (ny + 1) * (nz + 1)
    q = nz + 1
    elements = np.column_stack(
        [n0, n0 + p, n0 + p + q, n0 + q, n0 + 1, n0 + p + 1, n0 + p + q + 1, n0 + q + 1]
    )

    return nodes, elements


def _hex_to_tet(x_edges, y_edges, z_edges):
    """
    Tetrahedralise a rectangular prism defined by edge arrays.

    Each hex cell is split into 6 tetrahedra with the Kuhn (Freudenthal)
    decomposition: every tet shares the main diagonal from vertex 0 to
    vertex 6, one tet per monotone vertex path between them. The split
    is identical for every cell and puts the same face-local diagonal on
    opposite faces of each hex, so any two grids that share a boundary
    plane and transverse edges triangulate it identically — including
    across domain boundaries — with no cell-parity bookkeeping.

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

    # Hex vertices numbered:
    #   0 = (i,   j,   k  )   4 = (i,   j,   k+1)
    #   1 = (i+1, j,   k  )   5 = (i+1, j,   k+1)
    #   2 = (i+1, j+1, k  )   6 = (i+1, j+1, k+1)
    #   3 = (i,   j+1, k  )   7 = (i,   j+1, k+1)
    #
    # One tet per monotone path 0 -> 6, stepping +x/+y/+z in each of the
    # 6 possible orders (xyz, xzy, yxz, yzx, zxy, zyx).
    kuhn_tets = np.array(
        [
            (0, 1, 2, 6),
            (0, 1, 5, 6),
            (0, 3, 2, 6),
            (0, 3, 7, 6),
            (0, 4, 5, 6),
            (0, 4, 7, 6),
        ]
    )

    i, j, k = (
        grid.ravel()
        for grid in np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
    )
    n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    p = (ny + 1) * (nz + 1)
    q = nz + 1
    hex_verts = np.column_stack(
        [n0, n0 + p, n0 + p + q, n0 + q, n0 + 1, n0 + p + 1, n0 + p + q + 1, n0 + q + 1]
    )
    elements = hex_verts[:, kuhn_tets].reshape(-1, 4)

    return nodes, elements
