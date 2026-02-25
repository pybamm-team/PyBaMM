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

        # Boundary conditions
        bc_rhs = np.zeros(n)
        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            L, bc_rhs = self._apply_bcs_to_laplacian(submesh, L, bc_rhs, bcs)

        L_full = csr_matrix(kron(eye(repeats, dtype=np.float64), L))
        result = pybamm.Matrix(L_full) @ discretised_symbol

        if np.any(bc_rhs != 0):
            bc_rhs_full = np.tile(bc_rhs, repeats)
            result = result + pybamm.Vector(bc_rhs_full)

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

    def _apply_bcs_to_laplacian(self, submesh, L, bc_rhs, bcs):
        """Modify the Laplacian matrix and RHS for boundary conditions."""
        L = L.tolil()

        for side, (bc_value, bc_type) in bcs.items():
            face_tag = self._side_to_boundary_tag(side)
            if face_tag not in submesh.boundary_faces:
                continue

            face_indices = submesh.boundary_faces[face_tag]

            for fi in face_indices:
                cell = submesh.face_owner[fi]
                area = submesh.face_areas[fi]
                vol = submesh.cell_volumes[cell]

                face_centroid = submesh.face_centroids[fi]
                cell_centroid = submesh.cell_centroids[cell]
                d = np.linalg.norm(face_centroid - cell_centroid)

                coeff = area / d

                if bc_type == "Dirichlet":
                    bc_val = float(bc_value.evaluate())
                    L[cell, cell] -= coeff / vol
                    bc_rhs[cell] += coeff * bc_val / vol
                elif bc_type == "Neumann":
                    bc_val = float(bc_value.evaluate())
                    bc_rhs[cell] += bc_val * area / vol

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

        bc_vecs = [np.zeros(n) for _ in range(d)]
        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            G_components, bc_vecs = self._apply_bcs_to_gradient(
                submesh, G_components, bc_vecs, bcs
            )

        components = []
        for k in range(d):
            Gk = csr_matrix(kron(eye(repeats, dtype=np.float64), G_components[k]))
            comp = pybamm.Matrix(Gk) @ discretised_symbol
            if np.any(bc_vecs[k] != 0):
                bc_full = np.tile(bc_vecs[k], repeats)
                comp = comp + pybamm.Vector(bc_full)
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
        """Apply Dirichlet/Neumann BCs to gradient matrices."""
        d = submesh.dimension
        vol = submesh.cell_volumes

        for side, (bc_value, bc_type) in bcs.items():
            face_tag = self._side_to_boundary_tag(side)
            if face_tag not in submesh.boundary_faces:
                continue

            face_indices = submesh.boundary_faces[face_tag]

            if bc_type == "Dirichlet":
                bc_val = float(bc_value.evaluate())
                for fi in face_indices:
                    cell = submesh.face_owner[fi]
                    area = submesh.face_areas[fi]
                    normal = submesh.face_normals[fi]
                    face_c = submesh.face_centroids[fi]
                    cell_c = submesh.cell_centroids[cell]
                    dist = np.linalg.norm(face_c - cell_c)

                    # Replace zeroth-order boundary term u_owner * n * A / V
                    # with ghost-cell interpolation:
                    # u_f = (u_owner + u_ghost) / 2 where u_ghost = 2*bc_val - u_owner
                    # So u_f = bc_val, meaning:
                    #   remove existing contribution (u_owner * n * A / V)
                    #   add bc_val * n * A / V to RHS
                    # But the Green-Gauss matrix already has u_owner terms from
                    # _green_gauss_matrices. We need to zero out the boundary
                    # contribution and replace with the BC value.
                    # Simpler: the boundary face contribution to cell i is
                    #   G_k[cell, cell] gets n_k * A / V (from boundary term)
                    # For Dirichlet: u_f = bc_val, so contribution is
                    #   bc_val * n_k * A / V  (pure RHS, no matrix term)
                    # We need to subtract the existing matrix term and add RHS.
                    for k in range(d):
                        nk_A = normal[k] * area
                        # Remove the u_owner boundary term from the matrix
                        G_components[k] = G_components[k].tolil()
                        G_components[k][cell, cell] -= nk_A / vol[cell]
                        G_components[k] = csr_matrix(G_components[k])
                        # Add bc_val * n_k * A / V to the RHS
                        bc_vecs[k][cell] += bc_val * nk_A / vol[cell]

            elif bc_type == "Neumann":
                bc_val = float(bc_value.evaluate())
                for fi in face_indices:
                    cell = submesh.face_owner[fi]
                    area = submesh.face_areas[fi]
                    normal = submesh.face_normals[fi]
                    # Neumann: flux = bc_val at the face (in normal direction)
                    # The gradient's boundary contribution becomes:
                    #   (bc_val * dx + u_owner) * n_k * A / V
                    # For simplicity in the gradient, we keep the zeroth-order
                    # owner term and add the correction.
                    # Actually for Neumann BC on gradient, the face value is:
                    #   u_f = u_owner + bc_val * dist_to_face
                    # The extra contribution to the gradient is:
                    #   bc_val * dist * n_k * A / V
                    dist = np.linalg.norm(
                        submesh.face_centroids[fi] - submesh.cell_centroids[cell]
                    )
                    for k in range(d):
                        nk_A = normal[k] * area
                        bc_vecs[k][cell] += bc_val * dist * nk_A / vol[cell]

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

        disc_sv = getattr(discretised_symbol, "_disc_state_vector", None)
        if disc_sv is not None:
            L_bc, bc_rhs, D_bnd = self._div_boundary_correction(
                submesh, boundary_conditions, domain=domain
            )
            if D_bnd is not None:
                for k in range(d):
                    Dk_bnd = csr_matrix(kron(eye(repeats, dtype=np.float64), D_bnd[k]))
                    result = result - pybamm.Matrix(Dk_bnd) @ comps[k]
            if L_bc is not None:
                L_bc_full = csr_matrix(kron(eye(repeats, dtype=np.float64), L_bc))
                result = result + pybamm.Matrix(L_bc_full) @ disc_sv
            if np.any(bc_rhs != 0):
                bc_rhs_full = np.tile(bc_rhs, repeats)
                result = result + pybamm.Vector(bc_rhs_full)

        return result

    def _div_boundary_correction(self, submesh, boundary_conditions, domain=None):
        """Build boundary corrections for the divergence operator.

        When computing ``div(D * grad(u))``, the divergence matrices use
        cell-centered flux values at boundary faces, which is incorrect.
        This method returns:

        * ``L_bc`` – sparse matrix for TPFA Dirichlet correction on state vector
        * ``bc_rhs`` – constant vector for Dirichlet/Neumann RHS
        * ``D_bnd`` – list of sparse matrices (boundary-only divergence terms
          to subtract from the cell-centered approximation)

        The corrected divergence is::

            div(F) = sum_k D_k @ F_k - sum_k D_bnd_k @ F_k + L_bc @ u + bc_rhs
        """
        n = submesh.npts
        d = submesh.dimension
        L_bc = None
        bc_rhs = np.zeros(n)
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
                bc_val = float(bc_value.evaluate())

                for fi in face_indices:
                    cell = submesh.face_owner[fi]
                    area = submesh.face_areas[fi]
                    vol = submesh.cell_volumes[cell]
                    normal = submesh.face_normals[fi]

                    if D_bnd is None:
                        D_bnd = [csr_matrix((n, n)).tolil() for _ in range(d)]
                    for k in range(d):
                        D_bnd[k][cell, cell] += normal[k] * area / vol

                    if bc_type == "Dirichlet":
                        face_c = submesh.face_centroids[fi]
                        cell_c = submesh.cell_centroids[cell]
                        d_perp = np.linalg.norm(face_c - cell_c)
                        coeff = area / d_perp

                        if L_bc is None:
                            L_bc = csr_matrix((n, n)).tolil()
                        L_bc[cell, cell] -= coeff / vol
                        bc_rhs[cell] += coeff * bc_val / vol
                    elif bc_type == "Neumann":
                        bc_rhs[cell] += bc_val * area / vol

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

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        submesh = self.mesh[discretised_child.domain]
        n = submesh.npts
        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        side = symbol.side
        face_tag = self._side_to_boundary_tag(side)

        if face_tag not in submesh.boundary_faces:
            out = pybamm.Scalar(0)
            out.clear_domains()
            return out

        face_indices = submesh.boundary_faces[face_tag]
        n_bnd = len(face_indices)
        owners = submesh.face_owner[face_indices]

        if isinstance(symbol, pybamm.BoundaryGradient):
            # For boundary gradient, extrapolate gradient from cell center to face
            # using nearest cell value (zeroth-order) — improved when BCs available
            sub_matrix = csr_matrix(
                (np.ones(n_bnd), (np.arange(n_bnd), owners)),
                shape=(n_bnd, n),
            )
        else:
            # BoundaryValue: linear extrapolation from cell center to face
            # For unstructured meshes, use constant extrapolation (cell value)
            sub_matrix = csr_matrix(
                (np.ones(n_bnd), (np.arange(n_bnd), owners)),
                shape=(n_bnd, n),
            )

        mat = csr_matrix(kron(eye(repeats, dtype=np.float64), sub_matrix))
        bv_vector = pybamm.Matrix(mat)

        out = bv_vector @ discretised_child
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
