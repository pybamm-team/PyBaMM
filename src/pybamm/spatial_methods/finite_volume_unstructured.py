"""
Finite Volume spatial method for unstructured simplex meshes (2D triangles / 3D tets).

Dimension-agnostic: the same code path handles both 2D and 3D, with
dimension inferred from the mesh.  All operators are assembled from
face-cell connectivity as sparse matrices.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron

import pybamm


class FiniteVolumeUnstructured(pybamm.SpatialMethod):
    """
    Cell-centered finite volume method on unstructured simplex meshes.

    Supports triangles (2D) and tetrahedra (3D).  Operators:

    * **Laplacian** – Two-Point Flux Approximation (TPFA)
    * **Gradient** – Green-Gauss cell-centroid reconstruction
    * **Divergence** – face-flux summation (adjoint of gradient)
    * **Boundary conditions** – ghost-cell (Dirichlet) / direct injection (Neumann)

    Parameters
    ----------
    options : dict, optional
        Passed through to :class:`pybamm.SpatialMethod`.
    """

    def __init__(self, options=None):
        super().__init__(options)

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, mesh):
        super().build(mesh)
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts

    @staticmethod
    def _bc_contribution(n, n_bnd, owners, coeffs, bc_value):
        """Build a symbolic BC contribution vector.

        For scalar ``bc_value``: returns ``Vector(accumulated_coeffs) * bc_value``.
        For vector ``bc_value`` (length ``n_bnd``):
        returns ``Matrix(n, n_bnd) @ bc_value``.
        """
        is_scalar = isinstance(bc_value, pybamm.Scalar) or (
            hasattr(bc_value, "shape_for_testing")
            and bc_value.shape_for_testing == (1, 1)
        )
        if is_scalar:
            row = np.zeros(n)
            np.add.at(row, owners, coeffs)
            return pybamm.Vector(row) * bc_value
        else:
            M = csr_matrix((coeffs, (owners, np.arange(n_bnd))), shape=(n, n_bnd))
            return pybamm.Matrix(M) @ bc_value

    # ------------------------------------------------------------------
    # spatial_variable
    # ------------------------------------------------------------------

    def spatial_variable(self, symbol):
        symbol_mesh = self.mesh[symbol.domain]
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        direction = getattr(symbol, "direction", None)
        if direction is None:
            name = symbol.name
            if name.startswith("x"):
                col = 0
            elif name.startswith("y"):
                col = 1
            elif name.startswith("z"):
                col = symbol_mesh.dimension - 1
            else:
                col = 0
        else:
            col = {"lr": 0, "tb": symbol_mesh.dimension - 1, "fb": 1}.get(direction, 0)

        entries = np.tile(symbol_mesh.cell_centroids[:, col], repeats)
        return pybamm.Vector(entries, domains=symbol.domains)

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    def broadcast(self, symbol, domains, broadcast_type):
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
            identity = eye(symbol.shape[0])
            from scipy.sparse import vstack

            sec_size = self._get_auxiliary_domain_repeats(
                {"secondary": domains.get("secondary", [])}
            )
            matrix = vstack([identity for _ in range(sec_size)])
            out = pybamm.Matrix(matrix) @ symbol

        out.domains = domains.copy()
        return out

    # ==================================================================
    #  Core operators
    # ==================================================================

    # ------------------------------------------------------------------
    # Laplacian  (TPFA)
    # ------------------------------------------------------------------

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        domain = symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        L = self._tpfa_matrix(submesh)

        bc_rhs = pybamm.Vector(np.zeros(n))
        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            L, bc_rhs = self._apply_bcs_to_laplacian(submesh, L, bc_rhs, bcs)

        L_full = csr_matrix(kron(eye(repeats, dtype=np.float64), L))
        result = pybamm.Matrix(L_full) @ discretised_symbol + bc_rhs

        return result

    def _tpfa_matrix(self, submesh):
        """Assemble the TPFA Laplacian matrix for internal faces only.

        Includes the non-orthogonality correction: the coefficient for
        each face is scaled by ``(n_f · e_ij)`` where ``n_f`` is the
        outward face normal and ``e_ij`` is the unit vector from owner
        centroid to neighbor centroid.  On orthogonal meshes this factor
        is 1; on non-orthogonal meshes it corrects the first-order
        directional error.
        """
        n = submesh.npts
        n_int = submesh.n_internal_faces

        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]
        areas = submesh.face_areas[:n_int]
        normals = submesh.face_normals[:n_int]

        c_owner = submesh.cell_centroids[owner]
        c_neighbor = submesh.cell_centroids[neighbor]
        delta = c_neighbor - c_owner
        dist = np.linalg.norm(delta, axis=1)
        e_ij = delta / dist[:, np.newaxis]

        # Non-orthogonality correction: project normal onto centroid vector
        cos_theta = np.abs(np.sum(normals * e_ij, axis=1))

        coeff = areas * cos_theta / dist

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

        return csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))

    def div_D_grad(self, div_symbol, grad_child, disc_D, disc_u, boundary_conditions):
        """Discretise ``div(D * grad(u))`` as a single TPFA operation.

        Fully symbolic — works for both constant and state-dependent ``D``.
        Internal-face fluxes use arithmetic-mean interpolation of ``D`` to
        faces and a standard two-point difference for ``grad(u)``.
        """
        domain = div_symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        n_int = submesh.n_internal_faces
        repeats = self._get_auxiliary_domain_repeats(div_symbol.domains)
        vol = submesh.cell_volumes

        owner = submesh.face_owner[:n_int]
        neighbor = submesh.face_neighbor[:n_int]

        c_o = submesh.cell_centroids[owner]
        c_n = submesh.cell_centroids[neighbor]
        delta = c_n - c_o
        dist = np.linalg.norm(delta, axis=1)
        e_ij = delta / dist[:, np.newaxis]
        cos_theta = np.abs(np.sum(submesh.face_normals[:n_int] * e_ij, axis=1))
        geo = submesh.face_areas[:n_int] * cos_theta / dist

        # G (n_int x n): u_neighbor - u_owner per face
        G = csr_matrix(
            (
                np.concatenate([-np.ones(n_int), np.ones(n_int)]),
                (np.tile(np.arange(n_int), 2), np.concatenate([owner, neighbor])),
            ),
            shape=(n_int, n),
        )

        # W (n_int x n): arithmetic-mean D to faces
        W = csr_matrix(
            (
                np.full(2 * n_int, 0.5),
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

        G_f = csr_matrix(kron(eye(repeats, dtype=np.float64), G))
        W_f = csr_matrix(kron(eye(repeats, dtype=np.float64), W))
        S_f = csr_matrix(kron(eye(repeats, dtype=np.float64), S))
        geo_f = np.tile(geo, repeats)

        u_diff = pybamm.Matrix(G_f) @ disc_u
        is_scalar_D = isinstance(disc_D, pybamm.Scalar) or (
            hasattr(disc_D, "shape_for_testing") and disc_D.shape_for_testing == (1, 1)
        )
        if is_scalar_D:
            flux = disc_D * u_diff * pybamm.Vector(geo_f)
        else:
            D_face = pybamm.Matrix(W_f) @ disc_D
            flux = D_face * u_diff * pybamm.Vector(geo_f)
        result = pybamm.Matrix(S_f) @ flux

        # Boundary conditions
        bc_rhs = pybamm.Vector(np.zeros(n * repeats))
        if grad_child in boundary_conditions:
            bcs = boundary_conditions[grad_child]
            for side, (bc_value, bc_type) in bcs.items():
                face_tag = self._side_to_boundary_tag(side)
                if face_tag not in submesh.boundary_faces:
                    continue
                fi_arr = submesh.boundary_faces[face_tag]
                n_bnd = len(fi_arr)
                bnd_own = submesh.face_owner[fi_arr]

                E = csr_matrix(
                    (np.ones(n_bnd), (np.arange(n_bnd), bnd_own)),
                    shape=(n_bnd, n),
                )
                E_f = csr_matrix(kron(eye(repeats, dtype=np.float64), E))
                P = csr_matrix(
                    (np.ones(n_bnd), (bnd_own, np.arange(n_bnd))),
                    shape=(n, n_bnd),
                )
                P_f = csr_matrix(kron(eye(repeats, dtype=np.float64), P))
                D_bnd = disc_D if is_scalar_D else pybamm.Matrix(E_f) @ disc_D

                if bc_type == "Dirichlet":
                    geo_bnd = np.array(
                        [
                            submesh.face_areas[fi]
                            / np.linalg.norm(
                                submesh.face_centroids[fi]
                                - submesh.cell_centroids[bnd_own[j]]
                            )
                            / vol[bnd_own[j]]
                            for j, fi in enumerate(fi_arr)
                        ]
                    )
                    geo_bnd_f = np.tile(geo_bnd, repeats)

                    u_bnd = pybamm.Matrix(E_f) @ disc_u
                    bc_rhs = bc_rhs + pybamm.Matrix(P_f) @ (
                        D_bnd * (bc_value - u_bnd) * pybamm.Vector(geo_bnd_f)
                    )

                elif bc_type == "Neumann" and bc_value != pybamm.Scalar(0):
                    a_over_v = np.array(
                        [
                            submesh.face_areas[fi] / vol[bnd_own[j]]
                            for j, fi in enumerate(fi_arr)
                        ]
                    )
                    a_over_v_f = np.tile(a_over_v, repeats)
                    bc_rhs = bc_rhs + pybamm.Matrix(P_f) @ (
                        D_bnd * bc_value * pybamm.Vector(a_over_v_f)
                    )

        return result + bc_rhs

    def _apply_bcs_to_laplacian(self, submesh, L, bc_rhs, bcs):
        """Modify the Laplacian matrix and RHS for boundary conditions.

        ``bc_rhs`` is a pybamm expression (symbolic vector).
        """
        n = submesh.npts
        L = L.tolil()

        for side, (bc_value, bc_type) in bcs.items():
            face_tag = self._side_to_boundary_tag(side)
            if face_tag not in submesh.boundary_faces:
                continue

            face_indices = submesh.boundary_faces[face_tag]
            n_bnd = len(face_indices)
            owners = submesh.face_owner[face_indices]

            if bc_type == "Dirichlet":
                coeffs = np.empty(n_bnd)
                for j, fi in enumerate(face_indices):
                    cell = owners[j]
                    area = submesh.face_areas[fi]
                    vol = submesh.cell_volumes[cell]
                    d_perp = np.linalg.norm(
                        submesh.face_centroids[fi] - submesh.cell_centroids[cell]
                    )
                    coeff = area / d_perp
                    L[cell, cell] -= coeff / vol
                    coeffs[j] = coeff / vol
                bc_rhs = bc_rhs + self._bc_contribution(
                    n, n_bnd, owners, coeffs, bc_value
                )

            elif bc_type == "Neumann":
                coeffs = np.array(
                    [
                        submesh.face_areas[fi] / submesh.cell_volumes[owners[j]]
                        for j, fi in enumerate(face_indices)
                    ]
                )
                bc_rhs = bc_rhs + self._bc_contribution(
                    n, n_bnd, owners, coeffs, bc_value
                )

        return csr_matrix(L), bc_rhs

    @staticmethod
    def _side_to_boundary_tag(side):
        return {
            "left": "left",
            "right": "right",
            "top": "top",
            "bottom": "bottom",
            "front": "front",
            "back": "back",
        }.get(side, side)

    # ------------------------------------------------------------------
    # Gradient  (Green-Gauss)
    # ------------------------------------------------------------------

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        domain = symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        d = submesh.dimension
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        G_components = self._green_gauss_matrices(submesh)

        bc_vecs = [pybamm.Vector(np.zeros(n)) for _ in range(d)]
        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            G_components, bc_vecs = self._apply_bcs_to_gradient(
                submesh, G_components, bc_vecs, bcs
            )

        components = []
        for k in range(d):
            Gk = csr_matrix(kron(eye(repeats, dtype=np.float64), G_components[k]))
            comp = pybamm.Matrix(Gk) @ discretised_symbol + bc_vecs[k]
            components.append(comp)

        vf = pybamm.VectorField(*components)
        vf._disc_state_vector = discretised_symbol
        return vf

    def _green_gauss_matrices(self, submesh):
        """
        Build Green-Gauss gradient matrices G_k for k = 0..d-1.

        For each cell i, the gradient component k is:
            (grad u)_k,i = (1/V_i) * sum_f [u_f * n_k,f * A_f]

        where u_f is interpolated from owner/neighbor (distance-weighted
        for internal faces) or just the owner value (boundary faces).
        """
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

            # Contribution from owner side of internal face to cell "owner"
            #   G_k[owner, owner] += w_owner * nk_A / vol[owner]
            # Contribution from neighbor side of internal face to cell "owner"
            #   G_k[owner, neighbor] += w_neighbor * nk_A / vol[owner]
            # Same for the neighbor cell but with flipped normal
            #   G_k[neighbor, owner] -= w_owner * nk_A / vol[neighbor]
            #   G_k[neighbor, neighbor] -= w_neighbor * nk_A / vol[neighbor]

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

        return G

    def _apply_bcs_to_gradient(self, submesh, G_components, bc_vecs, bcs):
        """Apply Dirichlet/Neumann BCs to gradient matrices.

        ``bc_vecs`` is a list of pybamm expressions (one per spatial dimension).
        """
        n = submesh.npts
        d = submesh.dimension
        vol = submesh.cell_volumes

        for side, (bc_value, bc_type) in bcs.items():
            face_tag = self._side_to_boundary_tag(side)
            if face_tag not in submesh.boundary_faces:
                continue

            face_indices = submesh.boundary_faces[face_tag]
            n_bnd = len(face_indices)
            owners = submesh.face_owner[face_indices]

            if bc_type == "Dirichlet":
                for j, fi in enumerate(face_indices):
                    cell = owners[j]
                    normal = submesh.face_normals[fi]
                    area = submesh.face_areas[fi]
                    for k in range(d):
                        nk_A = normal[k] * area
                        G_components[k] = G_components[k].tolil()
                        G_components[k][cell, cell] -= nk_A / vol[cell]
                        G_components[k] = csr_matrix(G_components[k])

                for k in range(d):
                    coeffs = np.array(
                        [
                            submesh.face_normals[fi, k]
                            * submesh.face_areas[fi]
                            / vol[owners[j]]
                            for j, fi in enumerate(face_indices)
                        ]
                    )
                    bc_vecs[k] = bc_vecs[k] + self._bc_contribution(
                        n, n_bnd, owners, coeffs, bc_value
                    )

            elif bc_type == "Neumann":
                dists = np.linalg.norm(
                    submesh.face_centroids[face_indices]
                    - submesh.cell_centroids[owners],
                    axis=1,
                )
                for k in range(d):
                    coeffs = np.array(
                        [
                            dists[j]
                            * submesh.face_normals[fi, k]
                            * submesh.face_areas[fi]
                            / vol[owners[j]]
                            for j, fi in enumerate(face_indices)
                        ]
                    )
                    bc_vecs[k] = bc_vecs[k] + self._bc_contribution(
                        n, n_bnd, owners, coeffs, bc_value
                    )

        return G_components, bc_vecs

    # ------------------------------------------------------------------
    # Divergence
    # ------------------------------------------------------------------

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        domain = symbol.domain
        submesh = self.mesh[domain]
        n = submesh.npts
        d = submesh.dimension
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        if isinstance(discretised_symbol, pybamm.VectorField):
            comps = discretised_symbol._components
        elif isinstance(discretised_symbol, (list, tuple)):
            comps = list(discretised_symbol)
        else:
            raise TypeError(
                "FiniteVolumeUnstructured.divergence expects a VectorField or "
                f"list of {d} component arrays, got {type(discretised_symbol)}"
            )

        D_components = self._divergence_matrices(submesh)

        result = pybamm.Vector(np.zeros(n * repeats))
        for k in range(d):
            Dk = csr_matrix(kron(eye(repeats, dtype=np.float64), D_components[k]))
            result = result + pybamm.Matrix(Dk) @ comps[k]

        return result

    def _div_boundary_correction(self, submesh, boundary_conditions, domain=None):
        """Build boundary corrections for the divergence operator.

        When computing ``div(D * grad(u))``, the divergence matrices use
        cell-centered flux values at boundary faces, which is incorrect.
        This method returns:

        * ``L_bc`` – sparse matrix for TPFA Dirichlet correction on state vector
        * ``bc_rhs`` – symbolic pybamm expression for Dirichlet/Neumann RHS
        * ``D_bnd`` – list of sparse matrices (boundary-only divergence terms
          to subtract from the cell-centered approximation)

        The corrected divergence is::

            div(F) = sum_k D_k @ F_k - sum_k D_bnd_k @ F_k + L_bc @ u + bc_rhs
        """
        n = submesh.npts
        d = submesh.dimension
        L_bc = None
        bc_rhs = pybamm.Vector(np.zeros(n))
        D_bnd = None

        for var, bcs in boundary_conditions.items():
            if not hasattr(var, "domain"):
                continue
            if domain is not None and var.domain != domain:
                continue
            for side, (bc_value, bc_type) in bcs.items():
                face_tag = self._side_to_boundary_tag(side)
                if face_tag not in submesh.boundary_faces:
                    continue
                face_indices = submesh.boundary_faces[face_tag]
                n_bnd = len(face_indices)
                owners = submesh.face_owner[face_indices]
                areas = submesh.face_areas[face_indices]
                vols = submesh.cell_volumes[owners]

                if D_bnd is None:
                    D_bnd = [csr_matrix((n, n)).tolil() for _ in range(d)]
                for j, fi in enumerate(face_indices):
                    cell = owners[j]
                    normal = submesh.face_normals[fi]
                    for k in range(d):
                        D_bnd[k][cell, cell] += normal[k] * areas[j] / vols[j]

                if bc_type == "Dirichlet":
                    for j, fi in enumerate(face_indices):
                        cell = owners[j]
                        face_c = submesh.face_centroids[fi]
                        cell_c = submesh.cell_centroids[cell]
                        d_perp = np.linalg.norm(face_c - cell_c)
                        coeff = areas[j] / d_perp
                        if L_bc is None:
                            L_bc = csr_matrix((n, n)).tolil()
                        L_bc[cell, cell] -= coeff / vols[j]

                    coeffs = np.empty(n_bnd)
                    for j, fi in enumerate(face_indices):
                        face_c = submesh.face_centroids[fi]
                        cell_c = submesh.cell_centroids[owners[j]]
                        d_perp = np.linalg.norm(face_c - cell_c)
                        coeffs[j] = (areas[j] / d_perp) / vols[j]
                    bc_rhs = bc_rhs + self._bc_contribution(
                        n, n_bnd, owners, coeffs, bc_value
                    )

                elif bc_type == "Neumann":
                    coeffs = areas / vols
                    bc_rhs = bc_rhs + self._bc_contribution(
                        n, n_bnd, owners, coeffs, bc_value
                    )

        if L_bc is not None:
            L_bc = csr_matrix(L_bc)
        if D_bnd is not None:
            D_bnd = [csr_matrix(m) for m in D_bnd]
        return L_bc, bc_rhs, D_bnd

    def _divergence_matrices(self, submesh):
        """
        Build divergence matrices D_k for k = 0..d-1.

        For each cell i:
            (div F)_i = (1/V_i) * sum_f  F_k,f * n_k,f * A_f

        where F is the vector field components at cell centers. The face value
        is interpolated from owner/neighbor (same weights as gradient).
        """
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

        D = [csr_matrix((n, n)) for _ in range(d)]

        int_owner = owner[:n_int]
        int_neighbor = neighbor[:n_int]

        d_owner = np.linalg.norm(face_centroids[:n_int] - centroids[int_owner], axis=1)
        d_neighbor = np.linalg.norm(
            face_centroids[:n_int] - centroids[int_neighbor], axis=1
        )
        d_total = d_owner + d_neighbor
        w_owner = d_neighbor / d_total
        w_neighbor = d_owner / d_total

        for k in range(d):
            nk_A = normals[:n_int, k] * areas[:n_int]

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

            D[k] = D[k] + csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))

        # Boundary faces
        n_total = len(owner)
        bnd_indices = np.arange(n_int, n_total)
        if len(bnd_indices) > 0:
            bnd_owner = owner[bnd_indices]
            for k in range(d):
                nk_A = normals[bnd_indices, k] * areas[bnd_indices]
                D[k] = D[k] + csr_matrix(
                    coo_matrix(
                        (nk_A / vol[bnd_owner], (bnd_owner, bnd_owner)),
                        shape=(n, n),
                    )
                )

        return D

    # ------------------------------------------------------------------
    # gradient_squared  |grad u|^2
    # ------------------------------------------------------------------

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        result = None
        for comp in grad._components:
            sq = comp**2
            result = sq if result is None else result + sq
        return result

    # ------------------------------------------------------------------
    # Binary operator handling (scalar * VectorField, etc.)
    # ------------------------------------------------------------------

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
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
                        [disc_left._components[k], disc_right._components[k]]
                    )
                )
                for k in range(n)
            ]
            result = pybamm.VectorField(*new_comps)
            for src in (disc_left, disc_right):
                if hasattr(src, "_disc_state_vector"):
                    result._disc_state_vector = src._disc_state_vector
                    break
            return result

        return bin_op._binary_new_copy(disc_left, disc_right)

    # ------------------------------------------------------------------
    # Integral operators
    # ------------------------------------------------------------------

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable=None
    ):
        int_mat = self.definite_integral_matrix(child)
        repeats = self._get_auxiliary_domain_repeats(child.domains)
        mat = csr_matrix(kron(eye(repeats, dtype=np.float64), int_mat))
        return pybamm.Matrix(mat) @ discretised_child

    def definite_integral_matrix(self, child, vector_type="row", **kwargs):
        domain = child.domain
        if isinstance(domain, list):
            domain = tuple(domain)
        submesh = self.mesh[domain]
        vol = submesh.cell_volumes
        return csr_matrix(vol.reshape(1, -1))

    def boundary_integral(self, child, discretised_child, region):
        submesh = self.mesh[child.domain]
        face_tag = self._side_to_boundary_tag(region)
        repeats = self._get_auxiliary_domain_repeats(child.domains)

        if face_tag not in submesh.boundary_faces:
            return pybamm.Scalar(0)

        face_indices = submesh.boundary_faces[face_tag]
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
        submesh = self.mesh[discretised_child.domain]
        n = submesh.npts
        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        side = symbol.side

        if side in self._CORNER_SIDES:
            return self._corner_boundary_value(
                submesh, n, repeats, side, discretised_child
            )

        face_tag = self._side_to_boundary_tag(side)

        if face_tag not in submesh.boundary_faces:
            out = pybamm.Scalar(0)
            out.clear_domains()
            return out

        face_indices = submesh.boundary_faces[face_tag]
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
        """Extract value from the cell closest to a corner of the domain."""
        tb_side, lr_side = self._CORNER_SIDES[side]
        centroids = submesh.cell_centroids
        x_coords = centroids[:, 0]
        z_coords = centroids[:, -1]

        if lr_side == "right":
            target_x = x_coords.max()
        else:
            target_x = x_coords.min()
        if tb_side == "top":
            target_z = z_coords.max()
        else:
            target_z = z_coords.min()

        dists = (x_coords - target_x) ** 2 + (z_coords - target_z) ** 2
        cell_idx = int(np.argmin(dists))

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
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
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
    ):
        # Find the interface data between these two meshes.
        # The left_mesh should have interface_data keyed by
        # the right mesh's domain name (or vice versa).
        interface = None
        for data in left_mesh.interface_data.values():
            interface = data
            break

        if interface is None:
            for data in right_mesh.interface_data.values():
                interface = {
                    "left_cells": data["right_cells"],
                    "right_cells": data["left_cells"],
                    "face_areas": data["face_areas"],
                    "cell_distances": data["cell_distances"],
                }
                break

        if interface is None:
            raise ValueError(
                "No interface data found between the left and right meshes. "
                "Run compute_interface_data() during mesh construction."
            )

        n_faces = len(interface["left_cells"])
        n_left = left_mesh.npts
        n_right = right_mesh.npts

        left_sub = csr_matrix(
            (np.ones(n_faces), (np.arange(n_faces), interface["left_cells"])),
            shape=(n_faces, n_left),
        )
        right_sub = csr_matrix(
            (np.ones(n_faces), (np.arange(n_faces), interface["right_cells"])),
            shape=(n_faces, n_right),
        )

        inv_dx = diags(1.0 / interface["cell_distances"])
        left_weighted = inv_dx @ left_sub
        right_weighted = inv_dx @ right_sub

        left_mat = pybamm.Matrix(
            csr_matrix(kron(eye(repeats, dtype=np.float64), left_weighted))
        )
        right_mat = pybamm.Matrix(
            csr_matrix(kron(eye(repeats, dtype=np.float64), right_weighted))
        )

        dy_r = right_mat @ right_symbol_disc
        dy_r.clear_domains()
        dy_l = left_mat @ left_symbol_disc
        dy_l.clear_domains()

        return dy_r - dy_l

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
