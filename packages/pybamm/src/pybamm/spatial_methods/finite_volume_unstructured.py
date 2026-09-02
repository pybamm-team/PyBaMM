"""
Finite Volume spatial method for unstructured simplex meshes (2D triangles / 3D tets).

Dimension-agnostic: the same code path handles both 2D and 3D, with
dimension inferred from the mesh.  All operators are assembled from
face-cell connectivity as sparse matrices.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron
from scipy.spatial import cKDTree

import pybamm


class FiniteVolumeUnstructured(pybamm.SpatialMethod):
    """
    Cell-centered finite volume method on unstructured meshes.

    Supports triangles and quadrilaterals (2D), tetrahedra and hexahedra
    (3D).  Operators:

    * **Laplacian** – Two-Point Flux Approximation (TPFA) with an implicit
      non-orthogonal correction: the face normal is split as
      :math:`\\hat n = \\alpha \\hat e + \\mathbf{k}` along the unit
      centroid-to-centroid direction :math:`\\hat e`, so the normal
      derivative is :math:`\\alpha (u_j - u_i)/d + \\mathbf{k}\\cdot\\nabla
      u_f` with the cross term taken from the Green-Gauss gradient.  On
      orthogonal meshes :math:`\\mathbf{k} = 0` and this is plain TPFA.
    * **Gradient** – Green-Gauss cell-centroid reconstruction
    * **Divergence** – face-flux summation (adjoint of gradient)
    * **Boundary conditions** – ghost-cell (Dirichlet) / direct injection (Neumann)

    Neumann boundary values on the named axis sides (``"left"``/``"right"``,
    ``"front"``/``"back"``, ``"bottom"``/``"top"``) are coordinate-direction
    derivatives (:math:`\\partial u/\\partial x`, etc.), matching
    :class:`pybamm.FiniteVolume`; e.g. ``u = x`` takes value ``+1`` on both
    ``"left"`` and ``"right"``.  Values on any other face tag (Gmsh region
    names, ``"iface_*"``) are outward-normal derivatives
    :math:`\\partial u/\\partial n`.

    Parameters
    ----------
    options : dict, optional
        Passed through to :class:`pybamm.SpatialMethod`.  Additionally
        ``"non-orthogonal correction"`` selects the decomposition of the
        face normal: ``"over-relaxed"`` (default, :math:`\\alpha = 1/\\cos
        \\theta`, favouring diagonal dominance) or ``"minimum"``
        (:math:`\\alpha = \\cos\\theta`, the smallest cross term).  Both
        are exact on linear fields.
    """

    _CORRECTIONS = ("over-relaxed", "minimum")
    # Floor on cos(theta) in the over-relaxed weight (as in OpenFOAM): it
    # bounds alpha, and k is built from the same alpha so consistency holds.
    _COS_THETA_FLOOR = 0.05
    # Common CFD mesh-quality limit; beyond it the scheme stays consistent
    # but conditioning degrades.
    _NON_ORTHOGONALITY_WARNING_DEG = 70.0

    def __init__(self, options=None):
        super().__init__(options)
        self.options.setdefault("non-orthogonal correction", "over-relaxed")
        correction = self.options["non-orthogonal correction"]
        if correction not in self._CORRECTIONS:
            raise pybamm.OptionError(
                "'non-orthogonal correction' must be one of "
                f"{self._CORRECTIONS}, not {correction!r}"
            )

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, mesh):
        """See :meth:`pybamm.SpatialMethod.build`."""
        from pybamm.meshes.unstructured_submesh import UnstructuredSubMesh

        super().build(mesh)
        for dom in mesh:
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts
            sm = mesh[dom]
            if not isinstance(sm, UnstructuredSubMesh):
                continue
            name = dom[0] if isinstance(dom, tuple) else dom
            max_angle = self._face_geometry(sm)["max_angle_deg"]
            if max_angle > self._NON_ORTHOGONALITY_WARNING_DEG:
                pybamm.logger.warning(
                    f"Unstructured submesh for domain {name!r} has faces with "
                    f"{max_angle:.1f} degrees of non-orthogonality (angle "
                    "between the face normal and the centroid line). The "
                    "discretisation remains consistent but the linear systems "
                    "become poorly conditioned; consider improving the mesh."
                )
            # Tags come from the generator, not the constructor: a hand-built
            # mesh with none gets no BCs and is invisible to interface
            # discovery, so surface that before it fails downstream.
            if not sm.boundary_faces and len(sm.face_owner) > sm._boundary_face_start:
                pybamm.logger.warning(
                    f"Unstructured submesh for domain {name!r} has exterior "
                    "faces but no boundary tags: boundary conditions cannot "
                    "be applied and interface auto-discovery will not pair "
                    "it with neighboring domains. Tag it (e.g. "
                    "detect_box_boundaries() for axis-aligned boxes) or use "
                    "a mesh generator that supplies tags."
                )
        # Discover interfaces between all unstructured submesh pairs so
        # internal BCs work for arbitrary topology, not just 1D stacks.
        self._auto_compute_all_interfaces(mesh)

    # ------------------------------------------------------------------
    # interface auto-discovery (graph topology support)
    # ------------------------------------------------------------------

    @staticmethod
    def _interface_face_match(a_mesh, b_mesh, tol_factor=1e-3):
        """Return matched boundary-face index pairs between two submeshes.

        Boundary faces whose centroids coincide within
        :func:`pybamm.meshes.unstructured_submesh._geometric_tolerance`
        (``tol_factor`` of the smallest element edge) are paired.  Returns
        ``(a_idx, b_idx, matched)`` where ``matched`` is True iff at least
        one pair was found.
        """
        a_idx = (
            np.concatenate(list(a_mesh.boundary_faces.values()))
            if a_mesh.boundary_faces
            else np.array([], dtype=int)
        )
        b_idx = (
            np.concatenate(list(b_mesh.boundary_faces.values()))
            if b_mesh.boundary_faces
            else np.array([], dtype=int)
        )
        if (
            len(a_idx) == 0
            or len(b_idx) == 0
            # meshes of different spatial dimension can never share an interface
            or a_mesh.face_centroids.shape[1] != b_mesh.face_centroids.shape[1]
        ):
            return np.array([], dtype=int), np.array([], dtype=int), False
        from pybamm.meshes.unstructured_submesh import _geometric_tolerance

        a_c = a_mesh.face_centroids[a_idx]
        b_c = b_mesh.face_centroids[b_idx]
        # main's mesh module owns the geometric tolerance definition
        tol = _geometric_tolerance([a_mesh, b_mesh], rel=tol_factor)
        tree = cKDTree(b_c)
        d, j = tree.query(a_c, distance_upper_bound=tol)
        keep = np.isfinite(d)
        matched_b = j[keep]
        if len(np.unique(matched_b)) != len(matched_b):
            raise pybamm.GeometryError(
                f"Interface between meshes is not one-to-one: multiple faces "
                f"matched the same neighbor face within tolerance {tol:.2e}. "
                "The meshes are non-conforming at the interface."
            )
        return a_idx[keep], b_idx[matched_b], bool(keep.any())

    def _compute_pair_interface(self, a_mesh, b_mesh, a_name, b_name):
        """Populate ``interface_data`` and ``iface_<other>`` face buckets for
        a pair of submeshes that share a non-empty conformal interface.

        If either mesh already has an interface entry for the other (e.g. set
        up by 1D-stack auto-pairing in :class:`pybamm.Mesh` or by a manual
        ``compute_interface_data`` call), this method is a no-op so existing
        models keep their original face-tag scheme.
        """
        if b_name in a_mesh.interface_data or a_name in b_mesh.interface_data:
            return False
        a_match, b_match, ok = self._interface_face_match(a_mesh, b_mesh)
        if not ok:
            return False

        a_cells = a_mesh.face_owner[a_match]
        b_cells = b_mesh.face_owner[b_match]
        face_areas = a_mesh.face_areas[a_match]
        cell_distances = np.linalg.norm(
            b_mesh.cell_centroids[b_cells] - a_mesh.cell_centroids[a_cells],
            axis=1,
        )

        a_mesh.interface_data[b_name] = {
            "left_cells": a_cells,
            "right_cells": b_cells,
            "left_faces": a_match,
            "right_faces": b_match,
            "face_areas": face_areas,
            "cell_distances": cell_distances,
            "other_mesh": b_mesh,
        }
        b_mesh.interface_data[a_name] = {
            "left_cells": b_cells,
            "right_cells": a_cells,
            "left_faces": b_match,
            "right_faces": a_match,
            "face_areas": face_areas,
            "cell_distances": cell_distances,
            "other_mesh": a_mesh,
        }

        # Add new face-tag buckets for these interfaces.  Order matches
        # across both meshes so per-face BCs line up element-wise.
        a_iface_tag = f"iface_{b_name}"
        b_iface_tag = f"iface_{a_name}"
        a_mesh.boundary_faces[a_iface_tag] = a_match
        b_mesh.boundary_faces[b_iface_tag] = b_match

        # Remove interface faces from the axis-aligned buckets so external
        # BCs don't double-count them.
        a_match_set = {int(i) for i in a_match}
        b_match_set = {int(i) for i in b_match}
        for tag in list(a_mesh.boundary_faces.keys()):
            if tag.startswith("iface_"):
                continue
            keep = np.array(
                [int(i) not in a_match_set for i in a_mesh.boundary_faces[tag]],
                dtype=bool,
            )
            if keep.any():
                a_mesh.boundary_faces[tag] = a_mesh.boundary_faces[tag][keep]
            else:
                del a_mesh.boundary_faces[tag]
        for tag in list(b_mesh.boundary_faces.keys()):
            if tag.startswith("iface_"):
                continue
            keep = np.array(
                [int(i) not in b_match_set for i in b_mesh.boundary_faces[tag]],
                dtype=bool,
            )
            if keep.any():
                b_mesh.boundary_faces[tag] = b_mesh.boundary_faces[tag][keep]
            else:
                del b_mesh.boundary_faces[tag]
        return True

    def _auto_compute_all_interfaces(self, mesh):
        """Walk every pair of unstructured submeshes; pair faces where they
        coincide.  Replaces the 1D-stack adjacency assumption with arbitrary
        topology (star, tree, graph)."""
        from pybamm.meshes.unstructured_submesh import UnstructuredSubMesh

        domains = []
        for raw in mesh:
            name = raw[0] if isinstance(raw, tuple) else raw
            sm = mesh[raw]
            if isinstance(sm, UnstructuredSubMesh):
                domains.append((name, sm))
        seen = set()
        for (a, ma), (b, mb) in itertools.combinations(domains, 2):
            if ma is mb or (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            # returns False when the pair shares no conformal interface
            self._compute_pair_interface(ma, mb, a, b)

    # ------------------------------------------------------------------
    # internal BC assembly for arbitrary-topology Concatenation
    # ------------------------------------------------------------------

    def set_internal_bcs_for_concat(self, disc, var, children, outer_bcs):
        """Build internal BC dict for each ``Concatenation`` child by walking
        its mesh's ``interface_data`` graph instead of assuming consecutive
        1D-stack pairs.

        Returns ``None`` when no ``iface_<other>`` face buckets exist in any
        child mesh — that means there's no graph-discovered topology, so
        the caller should fall through to the legacy 1D-stack pairwise
        routine.

        For each child ``T_a`` on submesh ``mesh_a`` (graph case):
          - Pass through any user-supplied ``outer_bcs`` whose tag matches an
            external boundary tag present in ``mesh_a.boundary_faces``.
          - For each interface ``mesh_a ↔ mesh_b`` (one entry per neighbor
            in ``mesh_a.interface_data``), set ``iface_<b>`` to the
            discretised internal Neumann gradient between ``T_a`` and the
            matching child ``T_b``.
        """
        from pybamm.meshes.unstructured_submesh import UnstructuredSubMesh

        # Skip if no graph-discovered interfaces — caller falls back to the
        # default 1D-stack pairwise logic.
        has_iface = False
        for c in children:
            primary = c.domain[0]
            sm = self.mesh[primary]
            if isinstance(sm, UnstructuredSubMesh) and any(
                k.startswith("iface_") for k in sm.boundary_faces
            ):
                has_iface = True
                break
        if not has_iface:
            return None

        bcs_out = {}
        name_to_child = {c.domain[0]: c for c in children}
        for child in children:
            primary = child.domain[0]
            child_mesh = self.mesh[primary]
            if not isinstance(child_mesh, UnstructuredSubMesh):
                continue  # leave default handling for non-unstructured children
            bcs = {}
            for tag, bc_value in outer_bcs.items():
                if tag in child_mesh.boundary_faces:
                    bcs[tag] = bc_value
            for neighbor_name in child_mesh.interface_data:
                neighbor_child = name_to_child.get(neighbor_name)
                if neighbor_child is None:
                    continue
                if f"iface_{neighbor_name}" not in child_mesh.boundary_faces:
                    pybamm.logger.warning(
                        f"Domain {primary!r} has interface data for "
                        f"{neighbor_name!r} but no 'iface_{neighbor_name}' "
                        "face bucket; skipping the internal BC, so these "
                        "domains will not be coupled."
                    )
                    continue
                left_disc = disc.process_symbol(child)
                right_disc = disc.process_symbol(neighbor_child)
                neighbor_mesh = self.mesh[neighbor_name]
                # External conditions feed the interface gradient's cross
                # term; the interface faces themselves enter as cross rows.
                grad = self.internal_neumann_condition(
                    left_disc,
                    right_disc,
                    child_mesh,
                    neighbor_mesh,
                    left_bcs=self._external_bcs(child_mesh, outer_bcs),
                    right_bcs=self._external_bcs(neighbor_mesh, outer_bcs),
                )
                bcs[f"iface_{neighbor_name}"] = (grad, "Neumann")
            bcs_out[child] = bcs
        return bcs_out

    @staticmethod
    def _external_bcs(submesh, outer_bcs):
        """The entries of ``outer_bcs`` on this mesh's exterior face tags."""
        return {
            tag: bc
            for tag, bc in outer_bcs.items()
            if tag in submesh.boundary_faces and not tag.startswith("iface_")
        }

    @staticmethod
    def _bc_contribution(n, n_bnd, owners, coeffs, bc_value, repeats=1):
        """Build a symbolic BC contribution vector of size ``n * repeats``.

        For scalar ``bc_value``: returns ``Vector(accumulated_coeffs) * bc_value``.
        For vector ``bc_value``: returns ``Matrix @ bc_value``, where the value
        has one entry per boundary face (shared across auxiliary-domain
        repeats) or ``n_bnd * repeats`` entries (one per face per repeat).
        """
        is_scalar = isinstance(bc_value, pybamm.Scalar) or (
            hasattr(bc_value, "shape_for_testing")
            and bc_value.shape_for_testing == (1, 1)
        )
        if is_scalar:
            row = np.zeros(n)
            np.add.at(row, owners, coeffs)
            if repeats > 1:
                row = np.tile(row, repeats)
            return pybamm.Vector(row) * bc_value
        else:
            M = csr_matrix((coeffs, (owners, np.arange(n_bnd))), shape=(n, n_bnd))
            if repeats > 1:
                bc_shape = getattr(bc_value, "shape_for_testing", None)
                if bc_shape == (n_bnd * repeats, 1):
                    M = csr_matrix(kron(eye(repeats, dtype=np.float64), M))
                else:
                    M = csr_matrix(kron(np.ones((repeats, 1)), M))
            return pybamm.Matrix(M) @ bc_value

    @staticmethod
    def _tile_bc_value(bc_value, n_bnd, repeats):
        """Lift a BC value to ``n_bnd * repeats`` entries.

        Scalars and already-full values (``n_bnd * repeats`` entries) pass
        through; a per-face value (``n_bnd`` entries, shared across
        auxiliary-domain repeats) is tiled, matching :meth:`_bc_contribution`.
        """
        if repeats == 1:
            return bc_value
        is_scalar = isinstance(bc_value, pybamm.Scalar) or (
            hasattr(bc_value, "shape_for_testing")
            and bc_value.shape_for_testing == (1, 1)
        )
        if is_scalar or getattr(bc_value, "shape_for_testing", None) == (
            n_bnd * repeats,
            1,
        ):
            return bc_value
        tile = csr_matrix(kron(np.ones((repeats, 1)), eye(n_bnd, dtype=np.float64)))
        return pybamm.Matrix(tile) @ bc_value

    # ------------------------------------------------------------------
    # spatial_variable
    # ------------------------------------------------------------------

    def spatial_variable(self, symbol):
        """Return a vector of cell-centroid coordinates for ``symbol``'s
        direction (or its leading name token, e.g. ``x_n`` -> ``x``), tiled
        over auxiliary domains.  Raises :class:`pybamm.DomainError` rather
        than guessing when neither identifies a coordinate."""
        symbol_mesh = self.mesh[symbol.domain]
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)
        dim = symbol_mesh.dimension

        direction = getattr(symbol, "direction", None)
        if direction is not None:
            direction_cols = {"lr": 0, "fb": 1, "tb": dim - 1}
            if direction not in direction_cols or (direction == "fb" and dim == 2):
                valid = "'lr', 'tb'" if dim == 2 else "'lr', 'fb', 'tb'"
                raise pybamm.DomainError(
                    f"Unknown direction {direction!r} for spatial variable "
                    f"{symbol.name!r} on a {dim}D unstructured mesh; valid "
                    f"directions are {valid}."
                )
            col = direction_cols[direction]
        else:
            axis_name = symbol.name.split("_")[0]
            name_cols = {"x": 0, "y": 1, "z": dim - 1}
            if axis_name not in name_cols or (axis_name == "y" and dim == 2):
                valid = "'x'/'z'" if dim == 2 else "'x'/'y'/'z'"
                raise pybamm.DomainError(
                    f"Cannot infer a coordinate for spatial variable "
                    f"{symbol.name!r} on a {dim}D unstructured mesh; name it "
                    f"with a leading {valid} token (e.g. 'x_n') or set its "
                    "direction."
                )
            col = name_cols[axis_name]

        entries = np.tile(symbol_mesh.cell_centroids[:, col], repeats)
        return pybamm.Vector(entries, domains=symbol.domains)

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    def broadcast(self, symbol, domains, broadcast_type):
        """See :meth:`pybamm.SpatialMethod.broadcast`."""
        domain = domains["primary"]
        primary_pts = self.mesh[domain].npts
        aux_repeats = self._get_auxiliary_domain_repeats(domains)
        full_size = primary_pts * aux_repeats

        if broadcast_type.startswith("primary"):
            sub_vector = np.ones((primary_pts, 1))
            if symbol.shape_for_testing == ():
                out = symbol * pybamm.Vector(sub_vector)
            else:
                matrix = csr_matrix(kron(eye(symbol.shape_for_testing[0]), sub_vector))
                out = pybamm.Matrix(matrix) @ symbol
        elif broadcast_type.startswith("full"):
            out = symbol * pybamm.Vector(np.ones(full_size), domains=domains)
        else:
            from scipy.sparse import vstack

            # secondary/tertiary broadcast tiles the child by the size of the
            # new (slower-varying) dimension, matching SpatialMethod.broadcast
            if broadcast_type.startswith("secondary"):
                reps = self._get_auxiliary_domain_repeats(
                    {"secondary": domains.get("secondary", [])}
                )
            else:
                reps = self._get_auxiliary_domain_repeats(
                    {"tertiary": domains.get("tertiary", [])}
                )
            identity = eye(symbol.shape[0])
            matrix = vstack([identity for _ in range(reps)])
            out = pybamm.Matrix(matrix) @ symbol

        if out is symbol:
            # simplification can hand back the child itself (e.g. ones-vector
            # multiply); copy before stamping domains on a possibly shared node
            out = symbol.create_copy(perform_simplifications=False)
        out.domains = domains.copy()
        return out

    # ==================================================================
    #  Core operators
    # ==================================================================

    # ------------------------------------------------------------------
    # Laplacian  (TPFA)
    # ------------------------------------------------------------------

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """Laplacian ``Matrix @ discretised_symbol + bc_rhs``: the two-point
        flux plus the non-orthogonal cross term, which is built from the
        BC-aware Green-Gauss gradient so it stays fully implicit."""
        domain = symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        d = submesh.dimension
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        L = self._tpfa_matrix(submesh)
        K = self._cross_term_matrices(submesh)
        bcs = boundary_conditions.get(symbol, {})

        # The gradient is only assembled if some face actually needs a cross
        # term; on orthogonal meshes and boundaries the callable never fires.
        gradient_cache = []

        def gradient():
            if not gradient_cache:
                gradient_cache.append(
                    self._least_squares_gradient(submesh, bcs, repeats)
                )
            return gradient_cache[0]

        bc_rhs = pybamm.Vector(np.zeros(n * repeats))
        if bcs:
            L, bc_rhs = self._apply_bcs_to_laplacian(
                submesh, L, bc_rhs, bcs, repeats=repeats, gradient=gradient
            )

        if K is not None:
            G_components, grad_bc_vecs, _ = gradient()
            for k in range(d):
                L = L + K[k] @ G_components[k]
                if bcs:
                    K_full = csr_matrix(kron(eye(repeats, dtype=np.float64), K[k]))
                    bc_rhs = bc_rhs + pybamm.Matrix(K_full) @ grad_bc_vecs[k]
            L = csr_matrix(L)

        L_full = csr_matrix(kron(eye(repeats, dtype=np.float64), L))
        result = pybamm.Matrix(L_full) @ discretised_symbol + bc_rhs

        return result

    @staticmethod
    def _operator_cache(submesh):
        """Per-submesh cache for assembled operator matrices.

        Keyed on the face-owner connectivity so a cell reordering (e.g.
        ``optimize_ordering``) invalidates it. Cached matrices must never be
        mutated in place.
        """
        fingerprint = hash(submesh.face_owner.tobytes())
        cache = getattr(submesh, "_fv_operator_cache", None)
        if cache is None or cache.get("fingerprint") != fingerprint:
            cache = submesh._fv_operator_cache = {"fingerprint": fingerprint}
        return cache

    def _face_geometry(self, submesh):
        """Cached per-internal-face geometry shared by the TPFA operators.

        Returns a dict with the owner-to-neighbor centroid distance ``dist``
        and unit direction ``e_ij``, the signed ``cos_theta = n · e_ij``,
        the distance-weighted owner interpolation weight ``w_owner`` for
        face values, and the largest non-orthogonality angle in degrees.

        Raises
        ------
        pybamm.GeometryError
            If a face normal points away from the neighbor centroid: the
            two-point flux is undefined on such (inverted or non-star-shaped)
            cells.
        """
        cache = self._operator_cache(submesh)
        if "face_geometry" in cache:
            return cache["face_geometry"]
        n_int = submesh.n_internal_faces
        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]
        centroids = submesh.cell_centroids
        face_centroids = submesh.face_centroids[:n_int]

        delta = centroids[neighbor] - centroids[owner]
        dist = np.linalg.norm(delta, axis=1)
        e_ij = delta / dist[:, np.newaxis]
        cos_theta = np.sum(submesh.face_normals[:n_int] * e_ij, axis=1)
        if np.any(cos_theta <= 0):
            raise pybamm.GeometryError(
                f"{int(np.count_nonzero(cos_theta <= 0))} internal face(s) "
                "have a normal pointing away from the neighbor centroid "
                "(inverted or non-star-shaped cells), so the two-point flux "
                "is undefined there. Fix the mesh."
            )

        d_owner = np.linalg.norm(face_centroids - centroids[owner], axis=1)
        d_neighbor = np.linalg.norm(face_centroids - centroids[neighbor], axis=1)
        w_owner = d_neighbor / (d_owner + d_neighbor)

        max_angle = np.degrees(np.arccos(np.min(cos_theta))) if n_int else 0.0
        cache["face_geometry"] = {
            "dist": dist,
            "e_ij": e_ij,
            "cos_theta": cos_theta,
            "w_owner": w_owner,
            "max_angle_deg": float(max_angle),
        }
        return cache["face_geometry"]

    def _alpha(self, cos_theta):
        """Implicit weight of the two-point difference in ``n = alpha e + k``.

        Any ``alpha`` is consistent because ``k`` is built from the same
        value; the choice only sets how much flux the compact stencil
        carries versus the reconstructed-gradient cross term.
        """
        if self.options["non-orthogonal correction"] == "minimum":
            return cos_theta
        return 1.0 / np.maximum(cos_theta, self._COS_THETA_FLOOR)

    def _decomposition(self, submesh):
        """``(alpha, k)`` per internal face for ``n = alpha e_ij + k``."""
        geometry = self._face_geometry(submesh)
        alpha = self._alpha(geometry["cos_theta"])
        n_int = submesh.n_internal_faces
        k = submesh.face_normals[:n_int] - alpha[:, np.newaxis] * geometry["e_ij"]
        return alpha, k

    def _boundary_decomposition(self, submesh, faces):
        """``(dist, alpha, k)`` for boundary ``faces``, splitting the outward
        normal along the unit vector from the owner centroid to the face
        centroid: ``n = alpha e_b + k``.  ``dist * cos(theta)`` is the
        perpendicular distance, so ``alpha / dist`` is ``1 / (delta · n)``
        for the over-relaxed choice.
        """
        delta = (
            submesh.face_centroids[faces]
            - submesh.cell_centroids[submesh.face_owner[faces]]
        )
        dist = np.linalg.norm(delta, axis=1)
        e_b = delta / dist[:, np.newaxis]
        normals = submesh.face_normals[faces]
        alpha = self._alpha(np.sum(normals * e_b, axis=1))
        return dist, alpha, normals - alpha[:, np.newaxis] * e_b

    def _cross_term_matrices(self, submesh):
        """Assemble (or fetch the cached) matrices ``K_k`` mapping the cell
        gradient components to the cell divergence of the internal-face
        cross fluxes ``A_f k_f · grad(u)_f``, where ``grad(u)_f`` is the
        distance-weighted interpolation of the two cell gradients.

        Returns ``None`` when every internal face is orthogonal (``k = 0``),
        so orthogonal meshes pay nothing for the correction.
        """
        cache = self._operator_cache(submesh)
        key = ("cross", self.options["non-orthogonal correction"])
        if key in cache:
            return cache[key]
        n = submesh.npts
        n_int = submesh.n_internal_faces
        d = submesh.dimension
        _, k = self._decomposition(submesh)
        if n_int == 0 or np.max(np.abs(k)) < 1e-12:
            cache[key] = None
            return None

        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]
        areas = submesh.face_areas[:n_int]
        vol = submesh.cell_volumes
        w_owner = self._face_geometry(submesh)["w_owner"]

        face_rows = np.tile(np.arange(n_int), 2)
        both = np.concatenate([owner, neighbor])
        # P: cell gradient -> face gradient; S: face flux -> cell divergence
        # (+owner, -neighbor, so the cross flux is conservative by construction)
        P = csr_matrix(
            (np.concatenate([w_owner, 1.0 - w_owner]), (face_rows, both)),
            shape=(n_int, n),
        )
        S = csr_matrix(
            (
                np.concatenate([1.0 / vol[owner], -1.0 / vol[neighbor]]),
                (both, face_rows),
            ),
            shape=(n, n_int),
        )
        cache[key] = [csr_matrix(S @ diags(areas * k[:, kk]) @ P) for kk in range(d)]
        return cache[key]

    def _tpfa_matrix(self, submesh):
        """Assemble (or fetch the cached) two-point part of the Laplacian for
        internal faces only: the ``alpha (u_j - u_i) / d`` term of the
        decomposition ``n = alpha e_ij + k``.  :meth:`_cross_term_matrices`
        supplies the ``k · grad(u)_f`` remainder; on orthogonal meshes
        ``alpha = 1`` and this is the whole operator.
        """
        cache = self._operator_cache(submesh)
        key = ("tpfa", self.options["non-orthogonal correction"])
        if key in cache:
            return cache[key]
        n = submesh.npts
        n_int = submesh.n_internal_faces

        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]
        alpha, _ = self._decomposition(submesh)
        coeff = (
            submesh.face_areas[:n_int] * alpha / self._face_geometry(submesh)["dist"]
        )

        vol = submesh.cell_volumes

        rows = np.concatenate([owner, neighbor, owner, neighbor])
        cols = np.concatenate([neighbor, owner, owner, neighbor])
        data = np.concatenate(
            [
                coeff / vol[owner],
                coeff / vol[neighbor],
                -coeff / vol[owner],
                -coeff / vol[neighbor],
            ]
        )

        cache[key] = csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))
        return cache[key]

    def _div_D_grad_matrices(self, submesh):
        """Assemble (or fetch the cached) matrices for :meth:`div_D_grad`:
        ``G`` (two-point difference per internal face), ``W``
        (arithmetic-mean interpolation to faces), ``S`` (face flux to cell
        divergence), and the geometric factor ``geo`` per internal face.
        """
        cache = self._operator_cache(submesh)
        key = ("div_D_grad", self.options["non-orthogonal correction"])
        if key in cache:
            return cache[key]

        n = submesh.npts
        n_int = submesh.n_internal_faces
        vol = submesh.cell_volumes
        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]

        alpha, _ = self._decomposition(submesh)
        geo = submesh.face_areas[:n_int] * alpha / self._face_geometry(submesh)["dist"]

        # G (n_int x n): u_neighbor - u_owner per face
        G = csr_matrix(
            (
                np.concatenate([-np.ones(n_int), np.ones(n_int)]),
                (np.tile(np.arange(n_int), 2), np.concatenate([owner, neighbor])),
            ),
            shape=(n_int, n),
        )

        # W (n_int x n): distance-weighted interpolation of D to faces
        w_owner = self._face_geometry(submesh)["w_owner"]
        W = csr_matrix(
            (
                np.concatenate([w_owner, 1.0 - w_owner]),
                (np.tile(np.arange(n_int), 2), np.concatenate([owner, neighbor])),
            ),
            shape=(n_int, n),
        )

        # S (n x n_int): face flux -> cell divergence (+owner, -neighbor, /V)
        S = csr_matrix(
            (
                np.concatenate([1.0 / vol[owner], -1.0 / vol[neighbor]]),
                (np.concatenate([owner, neighbor]), np.tile(np.arange(n_int), 2)),
            ),
            shape=(n, n_int),
        )

        # C[k] (n_int x n): cell gradient component -> face cross flux
        # A_f k_f,k grad_k(u)_f, interpolated like D; None when orthogonal
        _, k_vec = self._decomposition(submesh)
        if np.max(np.abs(k_vec), initial=0.0) < 1e-12:
            C = None
        else:
            areas = submesh.face_areas[:n_int]
            C = [
                csr_matrix(diags(areas * k_vec[:, kk]) @ W)
                for kk in range(submesh.dimension)
            ]

        cache[key] = (G, W, S, geo, C)
        return cache[key]

    def div_D_grad(self, div_symbol, grad_child, disc_D, disc_u, boundary_conditions):
        """Discretise ``div(D * grad(u))`` as a single TPFA operation.

        Fully symbolic — works for both constant and state-dependent scalar
        ``D``. Internal-face fluxes use distance-weighted interpolation of
        ``D`` to faces and the two-point normal derivative plus its
        non-orthogonal cross term (see :meth:`_tpfa_matrix`).

        This method is only reached when the expression is written as
        ``div(D * grad(u))`` (a single product, matched syntactically during
        discretisation); other flux forms go through the generic
        :meth:`gradient`/:meth:`divergence` operators, which cannot apply
        boundary conditions conservatively and raise instead.
        """
        if isinstance(disc_D, pybamm.VectorField):
            raise pybamm.DiscretisationError(
                "Anisotropic (vector-valued) diffusion coefficients are not "
                "supported by the TPFA discretisation of div(D * grad(u))."
            )
        domain = div_symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        repeats = self._get_auxiliary_domain_repeats(div_symbol.domains)
        vol = submesh.cell_volumes

        G, W, S, geo, C = self._div_D_grad_matrices(submesh)
        bcs = boundary_conditions.get(grad_child, {})

        def lift(matrix):
            if repeats == 1:
                return matrix
            return csr_matrix(kron(eye(repeats, dtype=np.float64), matrix))

        def tile(values):
            return np.tile(values, repeats) if repeats > 1 else values

        # Cell gradient components, assembled lazily: only non-orthogonal
        # faces (internal or Dirichlet) need them for their cross term.
        gradient_cache = []

        def gradient():
            if not gradient_cache:
                G_grad, grad_bc, _ = self._least_squares_gradient(submesh, bcs, repeats)
                gradient_cache.append(
                    [
                        pybamm.Matrix(lift(G_grad[k])) @ disc_u + grad_bc[k]
                        for k in range(submesh.dimension)
                    ]
                )
            return gradient_cache[0]

        normal_grad = pybamm.Matrix(lift(G)) @ disc_u * pybamm.Vector(tile(geo))
        if C is not None:
            for k, grad_k in enumerate(gradient()):
                normal_grad = normal_grad + pybamm.Matrix(lift(C[k])) @ grad_k
        is_scalar_D = isinstance(disc_D, pybamm.Scalar) or (
            hasattr(disc_D, "shape_for_testing") and disc_D.shape_for_testing == (1, 1)
        )
        D_face = disc_D if is_scalar_D else pybamm.Matrix(lift(W)) @ disc_D
        result = pybamm.Matrix(lift(S)) @ (D_face * normal_grad)

        # Boundary conditions
        bc_rhs = pybamm.Vector(np.zeros(n * repeats))
        if bcs:
            for side, (bc_value, bc_type) in bcs.items():
                self._check_bc_type(bc_type)
                fi_arr = self._boundary_faces_for_side(submesh, side)
                n_bnd = len(fi_arr)
                bnd_own = submesh.face_owner[fi_arr]

                E = csr_matrix(
                    (np.ones(n_bnd), (np.arange(n_bnd), bnd_own)),
                    shape=(n_bnd, n),
                )
                P = csr_matrix(
                    (np.ones(n_bnd), (bnd_own, np.arange(n_bnd))),
                    shape=(n, n_bnd),
                )
                E_f, P_f = lift(E), lift(P)
                D_bnd = disc_D if is_scalar_D else pybamm.Matrix(E_f) @ disc_D
                bc_value = self._tile_bc_value(bc_value, n_bnd, repeats)
                a_over_v = submesh.face_areas[fi_arr] / vol[bnd_own]

                if bc_type == "Dirichlet":
                    dist, alpha, k_vec = self._boundary_decomposition(submesh, fi_arr)
                    u_bnd = pybamm.Matrix(E_f) @ disc_u
                    normal_grad_bnd = (bc_value - u_bnd) * pybamm.Vector(
                        tile(a_over_v * alpha / dist)
                    )
                    if np.max(np.abs(k_vec)) >= 1e-12:
                        for k, grad_k in enumerate(gradient()):
                            normal_grad_bnd = normal_grad_bnd + (
                                pybamm.Matrix(E_f) @ grad_k
                            ) * pybamm.Vector(tile(a_over_v * k_vec[:, k]))
                    bc_rhs = bc_rhs + pybamm.Matrix(P_f) @ (D_bnd * normal_grad_bnd)

                elif bc_type == "Neumann" and bc_value != pybamm.Scalar(0):
                    bc_rhs = bc_rhs + pybamm.Matrix(P_f) @ (
                        D_bnd
                        * bc_value
                        * pybamm.Vector(tile(self._neumann_sign(side) * a_over_v))
                    )

        return result + bc_rhs

    def _apply_bcs_to_laplacian(
        self, submesh, L, bc_rhs, bcs, repeats=1, gradient=None
    ):
        """Return the Laplacian matrix and RHS modified for boundary
        conditions.

        ``bc_rhs`` is a pybamm expression (symbolic vector of size
        ``npts * repeats``). ``L`` is not mutated (it may be cached).
        ``gradient`` is a zero-argument callable returning the
        ``(matrices, bc_vecs)`` of the cell gradient; it is only called for
        Dirichlet faces whose centroid direction is not normal to the face,
        which need the cross term ``A k · grad(u)``.  Without it those
        faces get the two-point term only.
        """
        n = submesh.npts
        d = submesh.dimension
        diag_correction = np.zeros(n)
        cross_diag = np.zeros((d, n))

        for side, (bc_value, bc_type) in bcs.items():
            self._check_bc_type(bc_type)
            face_indices = self._boundary_faces_for_side(submesh, side)
            n_bnd = len(face_indices)
            owners = submesh.face_owner[face_indices]
            a_over_v = submesh.face_areas[face_indices] / submesh.cell_volumes[owners]

            if bc_type == "Dirichlet":
                dist, alpha, k_vec = self._boundary_decomposition(submesh, face_indices)
                coeffs = a_over_v * alpha / dist
                np.add.at(diag_correction, owners, -coeffs)
                bc_rhs = bc_rhs + self._bc_contribution(
                    n, n_bnd, owners, coeffs, bc_value, repeats=repeats
                )
                if gradient is not None:
                    for k in range(d):
                        np.add.at(cross_diag[k], owners, a_over_v * k_vec[:, k])

            elif bc_type == "Neumann":
                coeffs = self._neumann_sign(side) * a_over_v
                bc_rhs = bc_rhs + self._bc_contribution(
                    n, n_bnd, owners, coeffs, bc_value, repeats=repeats
                )

        if np.any(diag_correction):
            L = csr_matrix(L + diags(diag_correction))
        if np.max(np.abs(cross_diag), initial=0.0) >= 1e-12:
            G_components, grad_bc_vecs, _ = gradient()
            for k in range(d):
                scale = diags(cross_diag[k])
                L = L + scale @ G_components[k]
                scale_full = csr_matrix(kron(eye(repeats, dtype=np.float64), scale))
                bc_rhs = bc_rhs + pybamm.Matrix(scale_full) @ grad_bc_vecs[k]
            L = csr_matrix(L)
        return L, bc_rhs

    @staticmethod
    def _boundary_faces_for_side(submesh, side):
        """Boundary-face indices for a BC side, raising when the tag is unknown.

        BC sides map directly onto ``submesh.boundary_faces`` keys; a missing
        key means the BC cannot be applied, so failing loudly here is what
        stops typos and interface-consumed sides from silently dropping BCs.
        """
        if side not in submesh.boundary_faces:
            raise pybamm.DiscretisationError(
                f"No boundary faces tagged {side!r} on this mesh (available "
                f"tags: {sorted(submesh.boundary_faces)}). The side may be "
                "misspelled, or its faces were absorbed into an internal "
                "interface by interface discovery."
            )
        return submesh.boundary_faces[side]

    @staticmethod
    def _check_bc_type(bc_type):
        if bc_type not in ("Dirichlet", "Neumann"):
            raise pybamm.DiscretisationError(
                f"boundary condition must be Dirichlet or Neumann, not {bc_type!r}"
            )

    # Named sides whose outward normal points along the negative coordinate
    # axis (see UnstructuredSubMesh._identify_boundary_faces).
    _NEGATIVE_NORMAL_SIDES = frozenset({"left", "front", "bottom"})

    @classmethod
    def _neumann_sign(cls, side):
        """Sign converting a Neumann boundary value to an outward-normal
        derivative.

        Named axis sides carry PyBaMM coordinate-direction values, so sides
        with a negative outward normal flip sign; any other face tag (Gmsh
        region names, ``iface_*``) is already outward-normal.
        """
        return -1.0 if side in cls._NEGATIVE_NORMAL_SIDES else 1.0

    # ------------------------------------------------------------------
    # Gradient  (Green-Gauss)
    # ------------------------------------------------------------------

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Least-squares cell-centroid gradient, returned as a
        :class:`pybamm.VectorField` with one component per dimension.

        Exact on linear fields for any cell shape: every face contributes
        one directional-derivative equation — towards the neighbour
        centroid (internal faces), towards the face centroid holding the
        prescribed value (Dirichlet), or the normal derivative itself
        (Neumann).  Boundary faces without a condition contribute nothing.
        """
        domain = symbol.domain
        submesh = self.mesh[domain]
        d = submesh.dimension
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        bcs = boundary_conditions.get(symbol, {})
        if bcs:
            missing = [tag for tag in submesh.boundary_faces if tag not in bcs]
            if missing:
                pybamm.logger.warning(
                    f"Gradient of {symbol.name!r}: boundary face buckets "
                    f"{missing} have no boundary condition and are fitted as "
                    "zero normal derivative (the operators' zero-flux "
                    "treatment), which is wrong if the field varies normal "
                    "to those boundaries."
                )
        G_components, bc_vecs, _ = self._least_squares_gradient(submesh, bcs, repeats)

        components = []
        for k in range(d):
            Gk = csr_matrix(kron(eye(repeats, dtype=np.float64), G_components[k]))
            comp = pybamm.Matrix(Gk) @ discretised_symbol + bc_vecs[k]
            components.append(comp)

        return pybamm.VectorField(*components)

    # Row kinds of the least-squares gradient fit
    _ROW_INTERNAL, _ROW_DIRICHLET, _ROW_NEUMANN, _ROW_NO_BC, _ROW_INTERFACE = range(5)

    def _face_bc_kinds(self, submesh, bcs, interface_faces=None):
        """Per-face row kind (see the ``_ROW_*`` constants)."""
        kinds = np.full(len(submesh.face_owner), self._ROW_NO_BC, dtype=int)
        kinds[: submesh.n_internal_faces] = self._ROW_INTERNAL
        for side, (_, bc_type) in bcs.items():
            self._check_bc_type(bc_type)
            faces = self._boundary_faces_for_side(submesh, side)
            kinds[faces] = (
                self._ROW_DIRICHLET if bc_type == "Dirichlet" else self._ROW_NEUMANN
            )
        if interface_faces is not None:
            kinds[interface_faces] = self._ROW_INTERFACE
        return kinds

    def _least_squares_matrices(self, submesh, bcs, interface=None):
        """Cached matrix part of the least-squares gradient for one BC layout.

        Each cell has the same number of faces ``m``, so the per-cell normal
        equations are solved in a batch.  Rows are unit-direction equations
        ``e · grad(u) = b``: ``e`` towards the neighbour centroid with
        ``b = (u_j - u_i) / dist`` (internal faces, and interface faces
        towards the other mesh's cell), towards the face centroid with
        ``b = (u_b - u_i) / dist`` (Dirichlet), or the outward normal with
        ``b`` the prescribed derivative (Neumann).  Boundary faces without a
        condition take ``b = 0``, matching the operators' zero-flux treatment
        of such faces.  Cells whose directions do not span the space get the
        minimum-norm fit via the pseudo-inverse.

        Parameters
        ----------
        submesh : UnstructuredSubMesh
        bcs : dict
            ``{side: (value, type)}`` boundary conditions.
        interface : dict, optional
            Cross-mesh rows: ``faces`` (this mesh's interface faces),
            ``other_cells`` and ``other_centroids`` (the paired cells of the
            other mesh), ``n_other`` and a hashable ``key`` for caching.

        Returns
        -------
        tuple
            ``(G, coeff, slot, length, G_cross)``: ``G[k]`` maps cell values
            to gradient component ``k`` and ``G_cross[k]`` (``None`` without
            an interface) maps the other mesh's values; ``coeff`` of shape
            ``(n, d, m)`` holds ``grad_k(cell) = sum_m coeff[cell, k, m]
            b_m``; ``slot[f]`` and ``length[f]`` are the row position within
            the owner cell and the row distance of face ``f``, used to place
            boundary values.
        """
        cache = self._operator_cache(submesh)
        signature = tuple(sorted((side, bc_type) for side, (_, bc_type) in bcs.items()))
        interface_key = None if interface is None else interface["key"]
        key = ("least_squares", signature, interface_key)
        if key in cache:
            return cache[key]

        n = submesh.npts
        d = submesh.dimension
        n_int = submesh.n_internal_faces
        n_faces = len(submesh.face_owner)
        centroids = submesh.cell_centroids
        interface_faces = None if interface is None else interface["faces"]
        kinds = self._face_bc_kinds(submesh, bcs, interface_faces)

        # Half-face rows: the owner side of every face, then the neighbour
        # side of internal faces, so row f (< n_faces) belongs to face f.
        row_face = np.concatenate([np.arange(n_faces), np.arange(n_int)])
        cell = np.concatenate([submesh.face_owner, submesh.face_neighbor[:n_int]])
        other = np.concatenate(
            [
                submesh.face_neighbor[:n_int],
                np.full(n_faces - n_int, -1),
                submesh.face_owner[:n_int],
            ]
        )
        row_kind = kinds[row_face]
        toward = np.where(
            (row_kind == self._ROW_INTERNAL)[:, np.newaxis],
            centroids[np.maximum(other, 0)],
            submesh.face_centroids[row_face],
        )
        if interface is not None:
            toward[interface_faces] = interface["other_centroids"]
            other[interface_faces] = interface["other_cells"]
        delta = toward - centroids[cell]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / length[:, np.newaxis]
        normal_rows = (row_kind == self._ROW_NEUMANN) | (row_kind == self._ROW_NO_BC)
        direction[normal_rows] = submesh.face_normals[row_face[normal_rows]]
        length[normal_rows] = 1.0

        counts = np.bincount(cell, minlength=n)
        if np.any(counts != counts[0]):
            raise pybamm.DiscretisationError(
                "Least-squares gradient needs every cell to have the same "
                "number of faces; the mesh connectivity is inconsistent."
            )
        m = int(counts[0])
        order = np.argsort(cell, kind="stable")
        dirs = direction[order].reshape(n, m, d)
        normal = np.einsum("nmi,nmj->nij", dirs, dirs)
        coeff = np.einsum("nij,nmj->nim", np.linalg.pinv(normal), dirs)
        slot = np.empty(len(cell), dtype=int)
        slot[order] = np.arange(len(cell)) % m

        def row_coefficients(rows, k):
            return coeff[cell[rows], k, slot[rows]] / length[rows]

        internal = np.nonzero(row_kind == self._ROW_INTERNAL)[0]
        dirichlet = np.nonzero(row_kind == self._ROW_DIRICHLET)[0]
        across = np.nonzero(row_kind == self._ROW_INTERFACE)[0]
        G = []
        G_cross = None if interface is None else []
        for k in range(d):
            c_int = row_coefficients(internal, k)
            c_dir = row_coefficients(dirichlet, k)
            c_across = row_coefficients(across, k)
            rows = np.concatenate(
                [cell[internal], cell[internal], cell[dirichlet], cell[across]]
            )
            cols = np.concatenate(
                [other[internal], cell[internal], cell[dirichlet], cell[across]]
            )
            data = np.concatenate([c_int, -c_int, -c_dir, -c_across])
            G.append(csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n))))
            if interface is not None:
                G_cross.append(
                    csr_matrix(
                        coo_matrix(
                            (c_across, (cell[across], other[across])),
                            shape=(n, interface["n_other"]),
                        )
                    )
                )

        cache[key] = (G, coeff, slot[:n_faces], length[:n_faces], G_cross)
        return cache[key]

    def _least_squares_gradient(self, submesh, bcs, repeats=1, interface=None):
        """``(matrices, bc_vecs, cross_matrices)`` of the least-squares
        gradient: component ``k`` is ``matrices[k] @ u + bc_vecs[k]``, plus
        ``cross_matrices[k] @ u_other`` when ``interface`` rows are given
        (sizes lifted by ``repeats`` for auxiliary domains)."""
        n = submesh.npts
        d = submesh.dimension
        G, coeff, slot, length, G_cross = self._least_squares_matrices(
            submesh, bcs, interface
        )
        bc_vecs = [pybamm.Vector(np.zeros(n * repeats)) for _ in range(d)]
        for side, (bc_value, bc_type) in bcs.items():
            faces = self._boundary_faces_for_side(submesh, side)
            owners = submesh.face_owner[faces]
            for k in range(d):
                coeffs = coeff[owners, k, slot[faces]]
                if bc_type == "Dirichlet":
                    coeffs = coeffs / length[faces]
                else:
                    coeffs = coeffs * self._neumann_sign(side)
                bc_vecs[k] = bc_vecs[k] + self._bc_contribution(
                    n, len(faces), owners, coeffs, bc_value, repeats=repeats
                )
        return G, bc_vecs, G_cross

    def _green_gauss_matrices(self, submesh):
        """
        Build (or fetch the cached) Green-Gauss gradient matrices G_k for
        k = 0..d-1.

        For each cell i, the gradient component k is:
            (grad u)_k,i = (1/V_i) * sum_f [u_f * n_k,f * A_f]

        where u_f is interpolated from owner/neighbor (distance-weighted
        for internal faces) or just the owner value (boundary faces).
        """
        cache = self._operator_cache(submesh)
        if "green_gauss" in cache:
            return cache["green_gauss"]
        n = submesh.npts
        d = submesh.dimension
        n_int = submesh.n_internal_faces

        owner = submesh.face_owner
        neighbor = submesh.face_neighbor
        normals = submesh.face_normals
        areas = submesh.face_areas
        vol = submesh.cell_volumes
        centroids = submesh.cell_centroids
        face_centroids = submesh.face_centroids

        G = [csr_matrix((n, n)) for _ in range(d)]

        # --- internal faces: distance-weighted interpolation ---
        int_owner = owner[:n_int]
        int_neighbor = neighbor[:n_int]

        d_owner = np.linalg.norm(face_centroids[:n_int] - centroids[int_owner], axis=1)
        d_neighbor = np.linalg.norm(
            face_centroids[:n_int] - centroids[int_neighbor], axis=1
        )
        d_total = d_owner + d_neighbor
        w_owner = d_neighbor / d_total  # weight for owner value
        w_neighbor = d_owner / d_total  # weight for neighbor value

        for k in range(d):
            nk_A = normals[:n_int, k] * areas[:n_int]

            # distance-weighted face value scatters to owner (+) and
            # neighbor (-), each divided by that cell's volume
            rows = np.concatenate([int_owner, int_owner, int_neighbor, int_neighbor])
            cols = np.concatenate([int_owner, int_neighbor, int_owner, int_neighbor])
            data = np.concatenate(
                [
                    w_owner * nk_A / vol[int_owner],
                    w_neighbor * nk_A / vol[int_owner],
                    -w_owner * nk_A / vol[int_neighbor],
                    -w_neighbor * nk_A / vol[int_neighbor],
                ]
            )

            G[k] = G[k] + csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))

        # --- boundary faces: u_f = u_owner (zeroth-order extrapolation) ---
        n_total = len(owner)
        bnd_indices = np.arange(n_int, n_total)
        if len(bnd_indices) > 0:
            bnd_owner = owner[bnd_indices]
            for k in range(d):
                nk_A = normals[bnd_indices, k] * areas[bnd_indices]
                rows = bnd_owner
                cols = bnd_owner
                data = nk_A / vol[bnd_owner]
                G[k] = G[k] + csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))

        cache["green_gauss"] = G
        return G

    # ------------------------------------------------------------------
    # Divergence
    # ------------------------------------------------------------------

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Face-flux divergence of a cell-centred vector field.

        Boundary faces use zeroth-order flux extrapolation, so fluxes built
        from BC-carrying gradients are rejected (see the raise below).
        """
        domain = symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        d = submesh.dimension
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # BC-bearing fluxes must use the div(D*grad(u)) TPFA intercept;
        # here the prescribed boundary flux would be silently ignored.
        bc_gradient_parents = [
            node.child
            for node in symbol.pre_order()
            if isinstance(node, pybamm.Gradient) and node.child in boundary_conditions
        ]
        if bc_gradient_parents:
            names = sorted({parent.name for parent in bc_gradient_parents})
            raise pybamm.DiscretisationError(
                f"Cannot discretise div of a general flux containing grad of "
                f"{names} on an unstructured mesh: the boundary conditions "
                "would be ignored and the result would not be conservative. "
                "Write the equation as div(D * grad(u)) (a single product) so "
                "the TPFA discretisation applies the boundary conditions."
            )

        if isinstance(discretised_symbol, pybamm.VectorField):
            comps = discretised_symbol.components
        elif isinstance(discretised_symbol, (list, tuple)):
            comps = list(discretised_symbol)
        else:
            raise pybamm.DiscretisationError(
                "FiniteVolumeUnstructured.divergence expects a VectorField or "
                f"list of {d} component arrays, got {type(discretised_symbol)}"
            )

        D_components = self._divergence_matrices(submesh)

        result = pybamm.Vector(np.zeros(n * repeats))
        for k in range(d):
            Dk = csr_matrix(kron(eye(repeats, dtype=np.float64), D_components[k]))
            result = result + pybamm.Matrix(Dk) @ comps[k]

        return result

    def _divergence_matrices(self, submesh):
        """Divergence matrices ``D_k``: ``(div F)_i = (1/V_i) sum_f F_k,f n_k,f A_f``.

        The face-value interpolation is identical to the Green-Gauss
        gradient's, so the two operators share one assembly.
        """
        return self._green_gauss_matrices(submesh)

    # ------------------------------------------------------------------
    # gradient_squared  |grad u|^2
    # ------------------------------------------------------------------

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        """Pointwise ``|grad u|^2`` via :meth:`gradient`."""
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        result = None
        for comp in grad.components:
            sq = comp**2
            result = sq if result is None else result + sq
        return result

    # ------------------------------------------------------------------
    # Binary operator handling (scalar * VectorField, etc.)
    # ------------------------------------------------------------------

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Apply a binary operator componentwise when either operand is a
        :class:`pybamm.VectorField`, lifting scalars to N components."""
        if isinstance(disc_left, pybamm.VectorField) or isinstance(
            disc_right, pybamm.VectorField
        ):
            if isinstance(disc_left, pybamm.VectorField) and isinstance(
                disc_right, pybamm.VectorField
            ):
                n = disc_left.n_components
            elif isinstance(disc_left, pybamm.VectorField):
                n = disc_left.n_components
                disc_right = pybamm.VectorField(*[disc_right] * n)
            else:
                n = disc_right.n_components
                disc_left = pybamm.VectorField(*[disc_left] * n)

            new_comps = [
                pybamm.simplify_if_constant(
                    bin_op.create_copy(
                        [disc_left.components[k], disc_right.components[k]]
                    )
                )
                for k in range(n)
            ]
            return pybamm.VectorField(*new_comps)

        return bin_op._binary_new_copy(disc_left, disc_right)

    # ------------------------------------------------------------------
    # Integral operators
    # ------------------------------------------------------------------

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable=None
    ):
        """Volume integral over the primary domain (cell-volume weights)."""
        int_mat = self.definite_integral_matrix(
            child, integration_dimension=integration_dimension
        )
        return int_mat @ discretised_child

    def definite_integral_matrix(
        self, child, vector_type="row", integration_dimension="primary"
    ):
        """Cell-volume weights of the primary domain as a
        :class:`pybamm.Matrix`, one block per auxiliary-domain repeat.

        Parameters
        ----------
        child : pybamm.Symbol
            The symbol being integrated.
        vector_type : str, optional
            ``"row"`` (default) or ``"column"``.
        integration_dimension : str, optional
            Only ``"primary"`` is supported: cells have no secondary
            structure to integrate over.

        Raises
        ------
        NotImplementedError
            For a non-primary ``integration_dimension``.
        """
        if integration_dimension != "primary":
            raise NotImplementedError(
                f"Integral in the {integration_dimension!r} dimension is not "
                "implemented on unstructured meshes; only the primary (cell) "
                "dimension can be integrated."
            )
        if vector_type not in ("row", "column"):
            raise pybamm.DiscretisationError(
                f"vector_type must be 'row' or 'column', not {vector_type!r}"
            )
        submesh = self.mesh[child.domain]
        repeats = self._get_auxiliary_domain_repeats(child.domains)
        shape = (1, -1) if vector_type == "row" else (-1, 1)
        block = csr_matrix(submesh.cell_volumes.reshape(shape))
        return pybamm.Matrix(csr_matrix(kron(eye(repeats, dtype=np.float64), block)))

    def boundary_integral(self, child, discretised_child, region):
        """Integral of the owner-cell values of ``child`` over the boundary
        faces of ``region`` (``"entire"`` = all exterior faces)."""
        submesh = self.mesh[child.domain]
        repeats = self._get_auxiliary_domain_repeats(child.domains)

        if region == "entire":
            # every exterior boundary face; iface_* buckets are internal
            # interfaces, not part of the domain boundary
            iface = [
                indices
                for tag, indices in submesh.boundary_faces.items()
                if tag.startswith("iface_")
            ]
            face_indices = np.setdiff1d(
                np.arange(submesh._boundary_face_start, len(submesh.face_owner)),
                np.concatenate(iface) if iface else np.array([], dtype=int),
            )
        else:
            face_indices = self._boundary_faces_for_side(submesh, region)
        n = submesh.npts

        owners = submesh.face_owner[face_indices]
        face_areas = submesh.face_areas[face_indices]

        row = np.zeros(n)
        np.add.at(row, owners, face_areas)
        mat = csr_matrix(row.reshape(1, -1))
        mat = csr_matrix(kron(eye(repeats, dtype=np.float64), mat))

        return pybamm.Matrix(mat) @ discretised_child

    # ------------------------------------------------------------------
    # boundary_value_or_flux
    # ------------------------------------------------------------------

    _CORNER_SIDES = {
        "top-right": ("top", "right"),
        "top-left": ("top", "left"),
        "bottom-right": ("bottom", "right"),
        "bottom-left": ("bottom", "left"),
    }

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """Owner-cell values on a boundary side (zeroth-order boundary
        value); corner sides return the single closest boundary cell."""
        if isinstance(symbol, pybamm.BoundaryGradient):
            raise NotImplementedError(
                "BoundaryGradient is not implemented for unstructured meshes; "
                "returning the boundary value instead would be silently wrong."
            )
        submesh = self.mesh[discretised_child.domain]
        n = submesh.npts
        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        side = symbol.side

        if side in self._CORNER_SIDES:
            return self._corner_boundary_value(
                submesh, n, repeats, side, discretised_child
            )

        face_indices = self._boundary_faces_for_side(submesh, side)
        n_bnd = len(face_indices)
        owners = submesh.face_owner[face_indices]

        sub_matrix = csr_matrix(
            (np.ones(n_bnd), (np.arange(n_bnd), owners)),
            shape=(n_bnd, n),
        )

        mat = csr_matrix(kron(eye(repeats, dtype=np.float64), sub_matrix))
        bv_vector = pybamm.Matrix(mat)

        out = bv_vector @ discretised_child
        out.clear_domains()
        return out

    def _corner_boundary_value(self, submesh, n, repeats, side, discretised_child):
        """Extract the value from the boundary cell closest to a corner of
        the (x, z) bounding box.

        Zeroth-order (cell value) regardless of the ``extrapolation`` option.
        Candidates are restricted to cells owning boundary faces on the two
        named sides, so interior cells of non-convex domains are never
        picked; in 3D, ties across y pick the lowest cell index.
        """
        tb_side, lr_side = self._CORNER_SIDES[side]
        centroids = submesh.cell_centroids

        # cells owning boundary faces on either named side (fall back to all
        # boundary-owner cells when a side bucket is missing)
        candidates = np.unique(
            np.concatenate(
                [
                    submesh.face_owner[submesh.boundary_faces[tag]]
                    for tag in (tb_side, lr_side)
                    if tag in submesh.boundary_faces
                ]
                or [submesh.face_owner[submesh._boundary_face_start :]]
            )
        )

        x_coords = centroids[candidates, 0]
        z_coords = centroids[candidates, -1]
        target_x = x_coords.max() if lr_side == "right" else x_coords.min()
        target_z = z_coords.max() if tb_side == "top" else z_coords.min()

        dists = (x_coords - target_x) ** 2 + (z_coords - target_z) ** 2
        cell_idx = int(candidates[np.argmin(dists)])

        sub_matrix = csr_matrix(
            (np.ones(1), (np.zeros(1, dtype=int), [cell_idx])),
            shape=(1, n),
        )
        mat = csr_matrix(kron(eye(repeats, dtype=np.float64), sub_matrix))
        out = pybamm.Matrix(mat) @ discretised_child
        out.clear_domains()
        return out

    # ------------------------------------------------------------------
    # internal_neumann_condition
    # ------------------------------------------------------------------

    def internal_neumann_condition(
        self,
        left_symbol_disc,
        right_symbol_disc,
        left_mesh,
        right_mesh,
        left_bcs=None,
        right_bcs=None,
    ):
        """Normal gradient across the interface between two submeshes, one
        value per interface face (outward from ``left_mesh``).

        On unstructured meshes this is the two-point difference plus the
        non-orthogonal cross term (see :meth:`_tpfa_matrix`), whose face
        gradient is fitted on both sides with the interface faces as
        cross-mesh rows.  ``left_bcs``/``right_bcs`` are the external
        boundary conditions of each side, used by that fit.
        """
        from pybamm.meshes.unstructured_submesh import UnstructuredSubMesh

        repeats = self._get_auxiliary_domain_repeats(left_symbol_disc.domains)

        if repeats != self._get_auxiliary_domain_repeats(right_symbol_disc.domains):
            raise pybamm.DomainError(
                "Number of secondary points in subdomains do not match"
            )

        if isinstance(left_mesh, UnstructuredSubMesh):
            return self._internal_neumann_unstructured(
                left_symbol_disc,
                right_symbol_disc,
                left_mesh,
                right_mesh,
                repeats,
                left_bcs or {},
                right_bcs or {},
            )
        else:
            return self._internal_neumann_structured(
                left_symbol_disc,
                right_symbol_disc,
                left_mesh,
                right_mesh,
                repeats,
            )

    def _internal_neumann_unstructured(
        self,
        left_symbol_disc,
        right_symbol_disc,
        left_mesh,
        right_mesh,
        repeats,
        left_bcs=None,
        right_bcs=None,
    ):
        left_bcs = left_bcs or {}
        right_bcs = right_bcs or {}
        # Find the interface_data entry that pairs left_mesh with right_mesh.
        # Each entry stores ``other_mesh`` so multi-neighbor topologies pick
        # the correct partner instead of grabbing the first dict value.
        interface = next(
            (
                data
                for data in left_mesh.interface_data.values()
                if data.get("other_mesh") is right_mesh
            ),
            None,
        )

        if interface is None:
            rev = next(
                (
                    data
                    for data in right_mesh.interface_data.values()
                    if data.get("other_mesh") is left_mesh
                ),
                None,
            )
            if rev is not None:
                interface = {
                    "left_cells": rev["right_cells"],
                    "right_cells": rev["left_cells"],
                    "left_faces": rev["right_faces"],
                    "right_faces": rev["left_faces"],
                    "face_areas": rev["face_areas"],
                    "cell_distances": rev["cell_distances"],
                }

        if interface is None:
            raise pybamm.DiscretisationError(
                "No interface data pairs these two unstructured meshes, so "
                "the internal gradient between them cannot be formed and the "
                "domains would be silently decoupled. Check that both meshes "
                "carry boundary tags (e.g. detect_box_boundaries() for "
                "axis-aligned boxes) so interface discovery can pair their "
                "faces."
            )

        left_cells = interface["left_cells"]
        right_cells = interface["right_cells"]
        left_faces = interface["left_faces"]
        n_faces = len(left_cells)
        d = left_mesh.dimension

        def lift(matrix):
            return pybamm.Matrix(
                csr_matrix(kron(eye(repeats, dtype=np.float64), matrix))
            )

        def without_domains(expr):
            expr.clear_domains()
            return expr

        def tile(values):
            return pybamm.Vector(np.tile(values, repeats))

        left_sub = csr_matrix(
            (np.ones(n_faces), (np.arange(n_faces), left_cells)),
            shape=(n_faces, left_mesh.npts),
        )
        right_sub = csr_matrix(
            (np.ones(n_faces), (np.arange(n_faces), right_cells)),
            shape=(n_faces, right_mesh.npts),
        )

        # n = alpha e + k with e the unit left-to-right centroid direction and
        # n the interface normal, outward from the left cells.
        normals = left_mesh.face_normals[left_faces]
        c_left = left_mesh.cell_centroids[left_cells]
        c_right = right_mesh.cell_centroids[right_cells]
        delta = c_right - c_left
        dist = np.linalg.norm(delta, axis=1)
        e_ij = delta / dist[:, np.newaxis]
        alpha = self._alpha(np.sum(normals * e_ij, axis=1))
        k_vec = normals - alpha[:, np.newaxis] * e_ij

        two_point = diags(alpha / dist)
        value = without_domains(
            lift(two_point @ right_sub) @ right_symbol_disc
        ) - without_domains(lift(two_point @ left_sub) @ left_symbol_disc)

        if np.max(np.abs(k_vec), initial=0.0) < 1e-12:
            return value

        face_centroids = left_mesh.face_centroids[left_faces]
        d_left = np.linalg.norm(face_centroids - c_left, axis=1)
        d_right = np.linalg.norm(face_centroids - c_right, axis=1)
        w_left = d_right / (d_left + d_right)

        def side_gradient(mesh, bcs, faces, other, other_cells, own_disc, other_disc):
            interface_rows = {
                "key": (id(other), hash(faces.tobytes())),
                "faces": faces,
                "other_cells": other_cells,
                "other_centroids": other.cell_centroids[other_cells],
                "n_other": other.npts,
            }
            G, bc_vecs, G_cross = self._least_squares_gradient(
                mesh, bcs, repeats, interface=interface_rows
            )
            return [
                without_domains(lift(G[k]) @ own_disc)
                + without_domains(lift(G_cross[k]) @ other_disc)
                + without_domains(bc_vecs[k])
                for k in range(d)
            ]

        grad_left = side_gradient(
            left_mesh,
            left_bcs,
            left_faces,
            right_mesh,
            right_cells,
            left_symbol_disc,
            right_symbol_disc,
        )
        grad_right = side_gradient(
            right_mesh,
            right_bcs,
            interface["right_faces"],
            left_mesh,
            left_cells,
            right_symbol_disc,
            left_symbol_disc,
        )
        for k in range(d):
            face_gradient = (
                lift(diags(w_left) @ left_sub) @ grad_left[k]
                + lift(diags(1.0 - w_left) @ right_sub) @ grad_right[k]
            )
            value = value + face_gradient * tile(k_vec[:, k])
        return value

    def _internal_neumann_structured(
        self,
        left_symbol_disc,
        right_symbol_disc,
        left_mesh,
        right_mesh,
        repeats,
    ):
        """Fallback for structured meshes (same logic as FiniteVolume)."""
        left_npts = left_mesh.npts
        right_npts = right_mesh.npts

        left_sub_matrix = np.zeros((1, left_npts))
        left_sub_matrix[0][left_npts - 1] = 1
        left_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(repeats, dtype=np.float64), left_sub_matrix))
        )

        right_sub_matrix = np.zeros((1, right_npts))
        right_sub_matrix[0][0] = 1
        right_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(repeats, dtype=np.float64), right_sub_matrix))
        )

        # structured fallback: 1D submeshes expose ``nodes``, not ``vertices``
        right_mesh_x = right_mesh.nodes[0]
        left_mesh_x = left_mesh.nodes[-1]
        dx = right_mesh_x - left_mesh_x

        dy_r = (right_matrix / dx) @ right_symbol_disc
        dy_r.clear_domains()
        dy_l = (left_matrix / dx) @ left_symbol_disc
        dy_l.clear_domains()

        return dy_r - dy_l

    # ------------------------------------------------------------------
    # concatenation
    # ------------------------------------------------------------------

    def concatenation(self, disc_children):
        """See :meth:`pybamm.SpatialMethod.concatenation`."""
        return pybamm.domain_concatenation(disc_children, self.mesh)

    # ------------------------------------------------------------------
    # Not implemented
    # ------------------------------------------------------------------

    def indefinite_integral(self, child, discretised_child, direction):
        raise NotImplementedError(
            "Indefinite integral is not supported on unstructured meshes. "
            "Use the direct PDE form instead."
        )

    def delta_function(self, symbol, discretised_symbol):
        raise NotImplementedError(
            "Delta function is not supported on unstructured meshes."
        )
