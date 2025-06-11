import numpy as np
from scipy.sparse import (
    block_diag,
    coo_matrix,
    csr_matrix,
    diags,
    eye,
    hstack,
    kron,
    lil_matrix,
    spdiags,
    vstack,
)

import pybamm


class FiniteVolume3D(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite volume method during
    discretisation.

    For broadcast and mass_matrix, we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the spatial method. The only option
        currently available is "extrapolation", which has options for "order" and "use_bcs".
        It sets the order separately for `pybamm.BoundaryValue` and `pybamm.BoundaryGradient`.
        Default is "linear" for the value and quadratic for the gradient.
    """

    def __init__(self, options=None):
        super().__init__(options)

    def build(self, mesh):
        super().build(mesh)

        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts

    def compute_spiral_metric(self, submesh):
        """
        Compute the custom metric for the spiral coordinate system.
        This function should return a scaling factor for each node or edge
        based on the local geometry of the spiral-wound structure.
        """
        r_nodes = submesh.nodes_x
        spiral_metric = 1 + 0.1 * np.sin(2 * np.pi * r_nodes / r_nodes.max())

        return spiral_metric

    def spatial_variable(self, symbol):
        """
        Creates a discretised spatial variable compatible with
        the FiniteVolume method.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        symbol_mesh = self.mesh[symbol.domain]
        symbol_direction = symbol.direction
        if symbol_mesh.dimension != 3:
            raise ValueError(f"Spatial variable {symbol} is not in 3D")
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        if symbol.evaluates_on_edges("primary"):
            X, Y, Z = np.meshgrid(
                symbol_mesh.edges_x,
                symbol_mesh.edges_y,
                symbol_mesh.edges_z,
                indexing="ij",
            )
            x = X.flatten(order="F")
            y = Y.flatten(order="F")
            z = Z.flatten(order="F")
        else:
            X, Y, Z = np.meshgrid(
                symbol_mesh.nodes_x,
                symbol_mesh.nodes_y,
                symbol_mesh.nodes_z,
                indexing="ij",
            )
            x = X.flatten(order="F")
            y = Y.flatten(order="F")
            z = Z.flatten(order="F")
        if symbol_direction == "x":
            entries = np.tile(x, repeats)
        elif symbol_direction == "y":
            entries = np.tile(y, repeats)
        elif symbol_direction == "z":
            entries = np.tile(z, repeats)
        else:
            raise ValueError(
                f"Symbol direction '{symbol_direction}' not supported for direct construction "
                "as a spatial variable vector. Discretise the variable directly."
            )

        return pybamm.Vector(entries, domains=symbol.domains)

    def _gradient(self, symbol, discretised_symbol, boundary_conditions, direction):
        """
        Gradient with a specific direction (x, y, or z)
        """
        domain = symbol.domain

        # Add Dirichlet boundary conditions, if defined
        if direction == "x":
            relevant_bcs = ["left", "right"]
        elif direction == "y":
            relevant_bcs = ["front", "back"]
        elif direction == "z":
            relevant_bcs = ["bottom", "top"]
        else:
            raise ValueError(f"Direction {direction} not supported")

        bcs = {}
        if symbol in boundary_conditions:
            bcs = {
                key: boundary_conditions[symbol][key]
                for key in relevant_bcs
                if key in boundary_conditions[symbol]
            }

        if any(bc[1] == "Dirichlet" for bc in bcs.values()):
            discretised_symbol, domain = self.add_ghost_nodes(
                symbol, discretised_symbol, bcs
            )

        grad_mat = self.gradient_matrix(domain, symbol.domains, direction)

        grad = grad_mat @ discretised_symbol
        grad.copy_domains(symbol)

        # Add Neumann boundary conditions, if defined
        if symbol in boundary_conditions:
            bcs = {
                key: boundary_conditions[symbol][key]
                for key in relevant_bcs
                if key in boundary_conditions[symbol]
            }
            if any(bc[1] == "Neumann" for bc in bcs.values()):
                grad = self.add_neumann_values(symbol, grad, bcs, domain)

        return grad

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        3D finite-volume gradient
        """
        grad_x = self._gradient(symbol, discretised_symbol, boundary_conditions, "x")
        grad_y = self._gradient(symbol, discretised_symbol, boundary_conditions, "y")
        grad_z = self._gradient(symbol, discretised_symbol, boundary_conditions, "z")

        grad = pybamm.VectorField3D(grad_x, grad_y, grad_z)
        return grad

    def gradient_matrix(self, domain, domains, direction):
        """
        Gradient matrix for finite volumes in 3D.
        Following PyBaMM convention: grad(y) = (y[1:] - y[:-1])/dx
        """
        if direction not in ["x", "y", "z"]:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'x', 'y', or 'z'"
            )

        submesh = self.mesh[domain]

        # Create matrix using submesh
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        # Use d_nodes for gradient (distance between node centers)
        e_x = 1 / submesh.d_nodes_x
        e_y = 1 / submesh.d_nodes_y
        e_z = 1 / submesh.d_nodes_z

        if direction == "x":
            # Gradient in x-direction: shape (n_x-1)*n_y*n_z x n_x*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes[:-1]  # Use left node for face weighting
                sub_matrix = diags(
                    [-e_x * r_weights, e_x * r_weights],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes[:-1] ** 2  # Use left node for face weighting
                sub_matrix = diags(
                    [-e_x * r_weights, e_x * r_weights],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                spiral_metric_faces = 0.5 * (spiral_metric[:-1] + spiral_metric[1:])
                if len(e_x) == 1:
                    e_x_broadcast = np.full(len(spiral_metric_faces), e_x[0])
                else:
                    e_x_broadcast = e_x[: len(spiral_metric_faces)]
                sub_matrix = diags(
                    [
                        -e_x_broadcast * spiral_metric_faces,
                        e_x_broadcast * spiral_metric_faces,
                    ],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            else:
                sub_matrix = diags([-e_x, e_x], [0, 1], shape=((n_x - 1), n_x))
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif direction == "y":
            # Gradient in y-direction: shape n_x*(n_y-1)*n_z x n_x*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = np.tile(r_nodes, n_y - 1)
                e_y_weighted = (
                    np.repeat(e_y, n_x) / r_weights[: len(np.repeat(e_y, n_x))]
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                theta_nodes = submesh.nodes_y
                # Spherical coordinate weighting
                r_weights = np.tile(r_nodes, n_y - 1)
                sin_weights = np.repeat(np.sin(theta_nodes[:-1]), n_x)
                e_y_weighted = np.repeat(e_y, n_x) / (
                    r_weights[: len(np.repeat(e_y, n_x))] * sin_weights
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            else:
                # Standard Cartesian case
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag((sub_matrix,) * n_z)

        elif direction == "z":
            # Gradient in z-direction: shape n_x*n_y*(n_z-1) x n_x*n_y*n_z
            if submesh.coord_sys in ["cylindrical polar", "spherical polar"]:
                # For cylindrical and spherical, z-direction is typically standard
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )
            else:
                # Standard Cartesian case
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)

        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        print(f"Gradient matrix creation for direction {direction}:")
        print(f"  sub_matrix.shape: {sub_matrix.shape}")
        print(f"  second_dim_repeats: {second_dim_repeats}")
        print(f"  final matrix.shape: {matrix.shape}")

        return pybamm.Matrix(matrix)

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator in 3D."""
        divergence_matrix_x = self.divergence_matrix(symbol.domains, "x")
        divergence_matrix_y = self.divergence_matrix(symbol.domains, "y")
        divergence_matrix_z = self.divergence_matrix(symbol.domains, "z")

        grad_x = discretised_symbol.x_field
        grad_y = discretised_symbol.y_field
        grad_z = discretised_symbol.z_field

        div_x = divergence_matrix_x @ grad_x
        div_y = divergence_matrix_y @ grad_y
        div_z = divergence_matrix_z @ grad_z

        return div_x + div_y + div_z

    def divergence_matrix(self, domains, direction):
        """
        Divergence matrix for a specific direction (x, y, or z)
        Following PyBaMM convention: div(N) = (N[1:] - N[:-1])/dx
        """

        submesh = self.mesh[domains["primary"]]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        if direction == "x":
            # Divergence in x-direction: shape n_x*n_y*n_z x (n_x-1)*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                sub_matrix = diags(
                    [-submesh.d_edges_x * r_nodes, submesh.d_edges_x * r_nodes],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x  # length n_x
                sub_matrix = diags(
                    [-submesh.d_edges_x * r_nodes**2, submesh.d_edges_x * r_nodes**2],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
            elif submesh.coord_sys == "spiral":
                spiral_metric_main = self.compute_spiral_metric(submesh)
                spiral_metric_super = self.compute_spiral_metric(submesh)
                sub_matrix = diags(
                    [
                        -submesh.d_edges_x * spiral_metric_main,
                        submesh.d_edges_x * spiral_metric_super,
                    ],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
            else:
                sub_matrix = diags(
                    [-submesh.d_edges_x, submesh.d_edges_x],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )

            sub_matrix = block_diag([sub_matrix] * (n_y * n_z), format="csr")

        elif direction == "y":
            e_y_values = submesh.d_edges_y
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x  # length n_x
                e_y_repeated = np.repeat(e_y_values, n_x)
                sub_matrix_plane = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y + 1)),
                )
            elif submesh.coord_sys == "spherical polar":
                e_y_repeated = np.repeat(e_y_values, n_x)
                sub_matrix_plane = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y + 1)),
                )
            else:
                e_y_repeated = (
                    np.repeat(
                        e_y_values, n_x + 1
                    )  # If e_y_values is scalar 1/dy or vector of length n_y
                )
                sub_matrix_plane = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y + 1)),
                )

            sub_matrix = block_diag([sub_matrix_plane] * n_z, format="csr")

        elif direction == "z":
            e_z_values = submesh.d_edges_z
            e_z_repeated = np.repeat(
                e_z_values, (n_x * n_y) + 1
            )  # If e_z_values is scalar 1/dz or vector of length n_z
            sub_matrix = diags(
                [-e_z_repeated, e_z_repeated],
                [0, n_x * n_y],
                shape=(n_x * n_y * n_z, n_x * n_y * (n_z + 1)),
            )

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)

        if not isinstance(sub_matrix, csr_matrix):
            sub_matrix = csr_matrix(sub_matrix)
        matrix = kron(eye(second_dim_repeats, format="csr"), sub_matrix, format="csr")

        return pybamm.Matrix(matrix)

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable
    ):
        """matrix-vector product to implement 3D integration operator."""
        integration_matrix = self.definite_integral_matrix(
            child,
            integration_dimension=integration_dimension,
            integration_variable=integration_variable,
        )

        if len(integration_variable) > 1:
            dirs = [v.direction for v in integration_variable]
            if len(set(dirs)) != len(dirs):
                raise ValueError("Integration variables must be in distinct directions")

            for _i, var in enumerate(integration_variable[1:], 1):
                direction = var.direction
                one_dimensional_matrix = self.one_dimensional_integral_matrix(
                    child, direction
                )
                if integration_matrix.shape[1] != one_dimensional_matrix.shape[0]:
                    repeats_needed = (
                        integration_matrix.shape[1] // one_dimensional_matrix.shape[0]
                    )
                    if repeats_needed > 1:
                        one_dimensional_matrix = kron(
                            eye(repeats_needed), one_dimensional_matrix
                        )
                integration_matrix = one_dimensional_matrix @ integration_matrix

        domains = child.domains
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        integration_matrix = kron(eye(second_dim_repeats), integration_matrix)
        return pybamm.Matrix(integration_matrix) @ discretised_child

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Laplacian operator in 3D, implemented as div(grad(.))
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        return self.divergence(grad, grad, boundary_conditions)

    def definite_integral_matrix(
        self,
        child,
        vector_type="row",
        integration_dimension="primary",
        integration_variable=None,
    ):
        """
        Create integration matrix for 3D finite volume method with support for all coordinate systems.
        """
        if integration_variable is None:
            raise ValueError("Integration variable must be provided for 3D integration")

        integration_direction = integration_variable[0].direction
        domains = child.domains
        domain = child.domains[integration_dimension]
        submesh = self.mesh[domain]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        if integration_dimension == "primary":
            submesh = self.mesh[domains["primary"]]
            if integration_direction == "x":
                d_edges = submesh.d_edges_x
                if submesh.coord_sys == "cylindrical polar":
                    r_nodes = submesh.nodes_x
                    weights = r_nodes  # r weighting for cylindrical
                elif submesh.coord_sys == "spherical polar":
                    r_nodes = submesh.nodes_x
                    weights = r_nodes**2  # r² weighting for spherical
                elif submesh.coord_sys == "spiral":
                    spiral_metric = self.compute_spiral_metric(submesh)
                    weights = spiral_metric
                else:
                    weights = np.ones(n_x)

                cols_list = []
                rows_list = []
                for k in range(n_z):
                    for j in range(n_y):
                        row_idx = k * n_y + j
                        col_start = k * n_x * n_y + j * n_x
                        cols_list.append(np.arange(col_start, col_start + n_x))
                        rows_list.append(np.full(n_x, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                data = np.tile(d_edges * weights, n_y * n_z)
                sub_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(n_y * n_z, n_x * n_y * n_z),
                )

            elif integration_direction == "y":
                d_edges = submesh.d_edges_y
                if submesh.coord_sys == "cylindrical polar":
                    weights = np.ones(n_y)
                elif submesh.coord_sys == "spherical polar":
                    theta_nodes = submesh.nodes_y
                    weights = np.sin(theta_nodes)
                elif submesh.coord_sys == "spiral":
                    weights = np.ones(n_y)
                else:
                    weights = np.ones(n_y)

                cols_list = []
                rows_list = []
                for k in range(n_z):
                    for i in range(n_x):
                        row_idx = k * n_x + i
                        cols_k = []
                        for j in range(n_y):
                            col_idx = k * n_x * n_y + j * n_x + i
                            cols_k.append(col_idx)
                        cols_list.append(cols_k)
                        rows_list.append(np.full(n_y, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                data = np.tile(d_edges * weights, n_x * n_z)
                sub_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(n_x * n_z, n_x * n_y * n_z),
                )

            elif integration_direction == "z":
                d_edges = submesh.d_edges_z
                if submesh.coord_sys == "cylindrical polar":
                    weights = np.ones(n_z)
                elif submesh.coord_sys == "spherical polar":
                    weights = np.ones(n_z)
                elif submesh.coord_sys == "spiral":
                    weights = np.ones(n_z)
                else:
                    weights = np.ones(n_z)

                cols_list = []
                rows_list = []
                for j in range(n_y):
                    for i in range(n_x):
                        row_idx = j * n_x + i
                        cols_j = []
                        for k in range(n_z):
                            col_idx = k * n_x * n_y + j * n_x + i
                            cols_j.append(col_idx)
                        cols_list.append(cols_j)
                        rows_list.append(np.full(n_z, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                data = np.tile(d_edges * weights, n_x * n_y)
                sub_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(n_x * n_y, n_x * n_y * n_z),
                )
        else:
            raise NotImplementedError(
                "Only primary integration dimension is implemented for 3D integration"
            )

        return sub_matrix

    def one_dimensional_integral_matrix(self, child, direction):
        """
        One-dimensional integral matrix for 3D finite volumes.
        """
        submesh = self.mesh[child.domain]
        domains = child.domains

        if direction == "x":
            d_edges = submesh.d_edges_y * submesh.d_edges_z
        elif direction == "y":
            d_edges = submesh.d_edges_x * submesh.d_edges_z
        elif direction == "z":
            d_edges = submesh.d_edges_x * submesh.d_edges_y

        # repeat matrix for each node in secondary dimensions
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = kron(eye(second_dim_repeats), d_edges)

        return matrix

    def boundary_integral(self, child, discretised_child, region):
        """
        Boundary integral operator for 3D.
        """
        symbol = pybamm.BoundaryValue(child, region)
        boundary_value = self.boundary_value_or_flux(symbol, discretised_child)

        if region in ["left", "right"]:
            direction = "x"
        elif region in ["front", "back"]:
            direction = "y"
        elif region in ["bottom", "top"]:
            direction = "z"
        else:
            raise ValueError(f"Region {region} not supported")

        integral_matrix = self.one_dimensional_integral_matrix(child, direction)
        domains = child.domains
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        integral_matrix = kron(eye(second_dim_repeats), integral_matrix)
        return pybamm.Matrix(integral_matrix) @ boundary_value

    def indefinite_integral(self, child, discretised_child, direction):
        """
        Indefinite integral operator for 3D.
        """
        if child.evaluates_on_edges("primary"):
            indefinite_integral_matrix = self.indefinite_integral_matrix_edges(
                child.domains, direction
            )
        else:
            submesh = self.mesh[child.domain]
            if submesh.coord_sys in ["cylindrical polar", "spherical polar", "spiral"]:
                raise NotImplementedError(
                    f"Indefinite integral on a {submesh.coord_sys} domain is not "
                    "implemented"
                )
            indefinite_integral_matrix = self.indefinite_integral_matrix_nodes(
                child.domains, direction
            )
        return indefinite_integral_matrix @ discretised_child

    def indefinite_integral_matrix_edges(self, domains, direction):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh edges in 3D.

        This follows the same logic as 1D but applies it to the primary dimension
        of the 3D mesh while keeping other dimensions fixed.
        """
        submesh = self.mesh[domains["primary"]]

        n_primary = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        du_n = submesh.d_nodes_x

        if direction == "forward":
            # Forward integration: cumulative sum from start
            du_entries = [du_n] * (n_primary - 1)
            offset = -np.arange(1, n_primary, 1)
            main_integral_matrix = spdiags(du_entries, offset, n_primary, n_primary - 1)
            bc_offset_matrix = lil_matrix((n_primary, n_primary - 1))
            bc_offset_matrix[:, 0] = du_n[0] / 2

        elif direction == "backward":
            # Backward integration: cumulative sum from end
            du_entries = [du_n] * (n_primary + 1)
            offset = np.arange(n_primary, -1, -1)
            main_integral_matrix = spdiags(du_entries, offset, n_primary, n_primary - 1)
            bc_offset_matrix = lil_matrix((n_primary, n_primary - 1))
            bc_offset_matrix[:, -1] = du_n[-1] / 2

        else:
            raise ValueError(
                f"Unknown direction: {direction}. Must be 'forward' or 'backward'"
            )

        # Combine main matrix and boundary condition offset
        sub_matrix_1d = main_integral_matrix + bc_offset_matrix

        zero_col = csr_matrix((n_primary, 1))
        sub_matrix_1d = hstack([zero_col, sub_matrix_1d, zero_col])

        sub_matrix_3d = kron(eye(n_y * n_z), sub_matrix_1d)

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_3d))

        if hasattr(submesh, "length"):
            matrix = matrix * submesh.length

        return pybamm.Matrix(matrix)

    def indefinite_integral_matrix_nodes(self, domains, direction):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh nodes in 3D.

        This is a straightforward cumulative sum along the primary dimension.
        """
        submesh = self.mesh[domains["primary"]]

        n_primary = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        du_n = submesh.d_edges_x

        du_entries = [du_n] * n_primary

        if direction == "forward":
            # Forward: cumulative sum from start to current
            offset = -np.arange(1, n_primary + 1, 1)
        elif direction == "backward":
            # Backward: cumulative sum from current to end
            offset = np.arange(n_primary - 1, -1, -1)
        else:
            raise ValueError(
                f"Unknown direction: {direction}. Must be 'forward' or 'backward'"
            )

        sub_matrix_1d = spdiags(du_entries, offset, n_primary + 1, n_primary)

        sub_matrix_3d = kron(eye(n_y * n_z), sub_matrix_1d)

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_3d))

        if hasattr(submesh, "length"):
            matrix = matrix * submesh.length

        return pybamm.Matrix(matrix)

    def delta_function(self, symbol, discretised_symbol):
        """
        3D delta function for boundary flux.
        Uses face area perpendicular to the integration direction, following 1D pattern.
        """
        submesh = self.mesh[symbol.domain]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

        if isinstance(symbol.side, tuple):
            coord_dir, face_side = symbol.side
        else:
            face_side = symbol.side
            coord_dir = "x"  # Default to x-direction

        if coord_dir == "x":
            x_idx = 0 if face_side == "left" else n_x - 1
            # Face area perpendicular to x (y-z area)
            face_area = (submesh.edges_y[-1] - submesh.edges_y[0]) * (
                submesh.edges_z[-1] - submesh.edges_z[0]
            )
            dx = submesh.d_edges_x[x_idx]
            rows = []
            for z_i in range(n_z):
                base_z = z_i * (n_x * n_y)
                for y_i in range(n_y):
                    rows.append(base_z + y_i * n_x + x_idx)
            scale = face_area / dx

        elif coord_dir == "y":
            y_idx = 0 if face_side == "left" else n_y - 1
            # Face area perpendicular to y (x-z area)
            face_area = (submesh.edges_x[-1] - submesh.edges_x[0]) * (
                submesh.edges_z[-1] - submesh.edges_z[0]
            )
            dy = submesh.d_edges_y[y_idx]
            rows = []
            for z_i in range(n_z):
                base_z = z_i * (n_x * n_y)
                for x_i in range(n_x):
                    rows.append(base_z + y_idx * n_x + x_i)
            scale = face_area / dy

        elif coord_dir == "z":
            z_idx = 0 if face_side == "left" else n_z - 1
            # Face area perpendicular to z (x-y area)
            face_area = (submesh.edges_x[-1] - submesh.edges_x[0]) * (
                submesh.edges_y[-1] - submesh.edges_y[0]
            )
            dz = submesh.d_edges_z[z_idx]
            rows = []
            for y_i in range(n_y):
                for x_i in range(n_x):
                    rows.append(z_idx * (n_x * n_y) + y_i * n_x + x_i)
            scale = face_area / dz
        else:
            raise ValueError(
                "symbol.side must be ('x','left'/'right'), ('y',...), or ('z',...) "
                "for tuple format, or 'left'/'right' for standard PyBaMM format"
            )

        rows = np.array(rows, dtype=int)
        cols = np.zeros_like(rows)
        data = np.ones_like(rows, dtype=float)
        sub_matrix = csr_matrix((data, (rows, cols)), shape=(n_x * n_y * n_z, 1))

        repeats = self._get_auxiliary_domain_repeats(symbol.domains)
        full_matrix = kron(eye(repeats), sub_matrix)

        delta_vec = pybamm.Matrix(scale * full_matrix) * discretised_symbol
        delta_vec.copy_domains(symbol)
        return delta_vec

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        """
        Internal Neumann conditions between two 3D symbols on adjacent subdomains.
        """
        left_npts = left_mesh.npts
        left_npts_x = left_mesh.npts_x
        left_npts_y = left_mesh.npts_y
        left_npts_z = left_mesh.npts_z

        right_npts = right_mesh.npts
        right_npts_x = right_mesh.npts_x
        right_npts_y = right_mesh.npts_y
        right_npts_z = right_mesh.npts_z

        second_dim_repeats = self._get_auxiliary_domain_repeats(
            left_symbol_disc.domains
        )

        if second_dim_repeats != self._get_auxiliary_domain_repeats(
            right_symbol_disc.domains
        ):
            raise pybamm.DomainError(
                "Number of secondary points in subdomains do not match"
            )

        left_sub_matrix = np.zeros((1, left_npts))
        for k in range(left_npts_z):
            for j in range(left_npts_y):
                idx = (
                    k * left_npts_x * left_npts_y + j * left_npts_x + (left_npts_x - 1)
                )
                left_sub_matrix[0, idx] = 1

        left_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), left_sub_matrix))
        )

        right_sub_matrix = np.zeros((1, right_npts))
        for k in range(right_npts_z):
            for j in range(right_npts_y):
                idx = k * right_npts_x * right_npts_y + j * right_npts_x + 0
                right_sub_matrix[0, idx] = 1

        right_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), right_sub_matrix))
        )

        right_mesh_x = right_mesh.nodes_x[0]
        left_mesh_x = left_mesh.nodes_x[-1]
        dx = right_mesh_x - left_mesh_x

        dy_r = (right_matrix / dx) @ right_symbol_disc
        dy_r.clear_domains()
        dy_l = (left_matrix / dx) @ left_symbol_disc
        dy_l.clear_domains()

        return dy_r - dy_l

    def add_ghost_nodes(self, symbol, discretised_symbol, bcs):
        """
        Add ghost nodes to a symbol.

        For Dirichlet bcs, for a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition and all other boundaries.

        For Neumann bcs no ghost nodes are added. Instead, the exact value provided
        by the boundary condition is used at the cell edge when calculating the
        gradient (see :meth:`pybamm.FiniteVolume3D.add_neumann_values`).

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_symbol : :class:`pybamm.Vector`
            Contains the discretised variable
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left", "right", "top", "bottom", "front", "back") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_symbol + bcs_vector`. When evaluated, this gives the
            discretised_symbol, with appropriate ghost nodes concatenated at each end.
        """
        domain = symbol.domain
        submesh = self.mesh[domain]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z
        n = submesh.npts
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # Catch if no boundary conditions are defined
        if (
            "left" not in bcs.keys()
            and "right" not in bcs.keys()
            and "top" not in bcs.keys()
            and "bottom" not in bcs.keys()
            and "front" not in bcs.keys()
            and "back" not in bcs.keys()
        ):
            raise ValueError(f"No boundary conditions have been provided for {symbol}")

        lbc_value, lbc_type = bcs.get("left", (None, None))
        rbc_value, rbc_type = bcs.get("right", (None, None))
        fbc_value, fbc_type = bcs.get("front", (None, None))
        bbc_value, bbc_type = bcs.get("back", (None, None))
        botbc_value, botbc_type = bcs.get("bottom", (None, None))
        tbc_value, tbc_type = bcs.get("top", (None, None))
        # Add ghost node(s) to domain where necessary and count number of
        # Dirichlet boundary conditions
        n_bcs = 0

        if fbc_type == "Dirichlet" and bbc_type != "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d + "_front ghost cell", d) for d in domain]
            else:
                domain = [domain + "_front ghost cell", domain]
            n_bcs += 1
        elif fbc_type != "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d, d + "_back ghost cell") for d in domain]
            else:
                domain = [domain, domain + "_back ghost cell"]
            n_bcs += 1
        elif fbc_type == "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [
                    (d + "_front ghost cell", d, d + "_back ghost cell") for d in domain
                ]
            else:
                domain = [
                    domain + "_front ghost cell",
                    domain,
                    domain + "_back ghost cell",
                ]
            n_bcs += 2

        if botbc_type == "Dirichlet" and tbc_type != "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d + "_bottom ghost cell", d) for d in domain]
            else:
                domain = [domain + "_bottom ghost cell", domain]
            n_bcs += 1
        elif botbc_type != "Dirichlet" and tbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d, d + "_top ghost cell") for d in domain]
            else:
                domain = [domain, domain + "_top ghost cell"]
            n_bcs += 1
        elif botbc_type == "Dirichlet" and tbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [
                    (d + "_bottom ghost cell", d, d + "_top ghost cell") for d in domain
                ]
            else:
                domain = [
                    domain + "_bottom ghost cell",
                    domain,
                    domain + "_top ghost cell",
                ]
            n_bcs += 2

        if lbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [domain[0] + "_left ghost cell", *domain]
            else:
                domain = [domain + "_left ghost cell", domain]
            n_bcs += 1
        if rbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [*domain, domain[-1] + "_right ghost cell"]
            else:
                domain = [domain, domain + "_right ghost cell"]
            n_bcs += 1

        print(f"Adding {n_bcs} ghost nodes to {symbol} in domain {domain}")

        # Calculate final dimensions with ALL ghost nodes
        final_n_x = (
            n_x
            + (1 if lbc_type == "Dirichlet" else 0)
            + (1 if rbc_type == "Dirichlet" else 0)
        )
        final_n_y = (
            n_y
            + (1 if fbc_type == "Dirichlet" else 0)
            + (1 if bbc_type == "Dirichlet" else 0)
        )
        final_n_z = (
            n_z
            + (1 if botbc_type == "Dirichlet" else 0)
            + (1 if tbc_type == "Dirichlet" else 0)
        )
        final_size = final_n_x * final_n_y * final_n_z * second_dim_repeats

        # X-direction boundaries (follow 2D pattern exactly)
        if lbc_type == "Dirichlet":
            lbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(final_n_x, 1))
            lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
            lbc_matrix = vstack([lbc_matrix] * (final_n_y * final_n_z))

            if lbc_value.evaluates_to_number():
                left_ghost_constant = (
                    2 * lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                left_ghost_constant = 2 * lbc_value
            lbc_vector = pybamm.Matrix(lbc_matrix) @ left_ghost_constant
        else:
            lbc_vector = pybamm.Vector(np.zeros(final_size))

        if rbc_type == "Dirichlet":
            rbc_sub_matrix = coo_matrix(
                ([1], ([final_n_x - 1], [0])), shape=(final_n_x, 1)
            )
            rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
            rbc_matrix = vstack([rbc_matrix] * (final_n_y * final_n_z))

            if rbc_value.evaluates_to_number():
                right_ghost_constant = (
                    2 * rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                right_ghost_constant = 2 * rbc_value
            rbc_vector = pybamm.Matrix(rbc_matrix) @ right_ghost_constant
        else:
            rbc_vector = pybamm.Vector(np.zeros(final_size))

        # Y-direction boundaries
        if fbc_type == "Dirichlet":
            fbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(final_n_y, 1))
            fbc_matrix = csr_matrix(kron(eye(second_dim_repeats), fbc_sub_matrix))
            fbc_matrix = vstack([fbc_matrix] * (final_n_x * final_n_z))

            if fbc_value.evaluates_to_number():
                front_ghost_constant = (
                    2 * fbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                front_ghost_constant = 2 * fbc_value
            fbc_vector = pybamm.Matrix(fbc_matrix) @ front_ghost_constant
        else:
            fbc_vector = pybamm.Vector(np.zeros(final_size))

        if bbc_type == "Dirichlet":
            bbc_sub_matrix = coo_matrix(
                ([1], ([final_n_y - 1], [0])), shape=(final_n_y, 1)
            )
            bbc_matrix = csr_matrix(kron(eye(second_dim_repeats), bbc_sub_matrix))
            bbc_matrix = vstack([bbc_matrix] * (final_n_x * final_n_z))

            if bbc_value.evaluates_to_number():
                back_ghost_constant = (
                    2 * bbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                back_ghost_constant = 2 * bbc_value
            bbc_vector = pybamm.Matrix(bbc_matrix) @ back_ghost_constant
        else:
            bbc_vector = pybamm.Vector(np.zeros(final_size))

        # Z-direction boundaries
        if botbc_type == "Dirichlet":
            botbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(final_n_z, 1))
            botbc_matrix = csr_matrix(kron(eye(second_dim_repeats), botbc_sub_matrix))
            botbc_matrix = vstack([botbc_matrix] * (final_n_x * final_n_y))

            if botbc_value.evaluates_to_number():
                bottom_ghost_constant = (
                    2 * botbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                bottom_ghost_constant = 2 * botbc_value
            botbc_vector = pybamm.Matrix(botbc_matrix) @ bottom_ghost_constant
        else:
            botbc_vector = pybamm.Vector(np.zeros(final_size))

        if tbc_type == "Dirichlet":
            tbc_sub_matrix = coo_matrix(
                ([1], ([final_n_z - 1], [0])), shape=(final_n_z, 1)
            )
            tbc_matrix = csr_matrix(kron(eye(second_dim_repeats), tbc_sub_matrix))
            tbc_matrix = vstack([tbc_matrix] * (final_n_x * final_n_y))

            if tbc_value.evaluates_to_number():
                top_ghost_constant = (
                    2 * tbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                top_ghost_constant = 2 * tbc_value
            tbc_vector = pybamm.Matrix(tbc_matrix) @ top_ghost_constant
        else:
            tbc_vector = pybamm.Vector(np.zeros(final_size))

        bcs_vector = (
            lbc_vector
            + rbc_vector
            + fbc_vector
            + bbc_vector
            + botbc_vector
            + tbc_vector
        )
        print(f"  bcs_vector.shape: {bcs_vector.shape}")
        bcs_vector.copy_domains(discretised_symbol)

        if lbc_type == "Dirichlet":
            left_ghost_vector = coo_matrix(([-1], ([0], [0])), shape=(1, n_x))
            print(f"  left_ghost_vector.shape: {left_ghost_vector.shape}")
        else:
            left_ghost_vector = None
        if rbc_type == "Dirichlet":
            right_ghost_vector = coo_matrix(([-1], ([0], [n_x - 1])), shape=(1, n_x))
            print(f"  right_ghost_vector.shape: {right_ghost_vector.shape}")
        else:
            right_ghost_vector = None

        if fbc_type == "Dirichlet":
            row_indices = np.arange(0, n_x * n_z)
            col_indices = np.arange(0, n_x * n_z)
            front_ghost_vector = coo_matrix(
                (-np.ones(n_x * n_z), (row_indices, col_indices)), shape=(n_x * n_z, n)
            )
        else:
            front_ghost_vector = None
        if bbc_type == "Dirichlet":
            row_indices = np.arange(0, n_x * n_z)
            col_indices = np.arange(n - n_x * n_z, n)
            back_ghost_vector = coo_matrix(
                (-np.ones(n_x * n_z), (row_indices, col_indices)), shape=(n_x * n_z, n)
            )
        else:
            back_ghost_vector = None

        if botbc_type == "Dirichlet":
            row_indices = np.arange(0, n_x * n_y)
            col_indices = np.arange(0, n_x * n_y)
            bottom_ghost_vector = coo_matrix(
                (-np.ones(n_x * n_y), (row_indices, col_indices)), shape=(n_x * n_y, n)
            )
        else:
            bottom_ghost_vector = None
        if tbc_type == "Dirichlet":
            row_indices = np.arange(0, n_x * n_y)
            col_indices = np.arange(n - n_x * n_y, n)
            top_ghost_vector = coo_matrix(
                (-np.ones(n_x * n_y), (row_indices, col_indices)), shape=(n_x * n_y, n)
            )
        else:
            top_ghost_vector = None

        if lbc_type == "Dirichlet" or rbc_type == "Dirichlet":
            sub_matrix = vstack(
                [
                    mat
                    for mat in [left_ghost_vector, eye(n_x), right_ghost_vector]
                    if mat is not None
                ]
            )
            sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
        else:
            matrix_parts = []
            if left_ghost_vector is not None:
                matrix_parts.append(left_ghost_vector)
            if front_ghost_vector is not None:
                matrix_parts.append(front_ghost_vector)
            if bottom_ghost_vector is not None:
                matrix_parts.append(bottom_ghost_vector)
            matrix_parts.append(eye(n))
            if top_ghost_vector is not None:
                matrix_parts.append(top_ghost_vector)
            if back_ghost_vector is not None:
                matrix_parts.append(back_ghost_vector)
            if right_ghost_vector is not None:
                matrix_parts.append(right_ghost_vector)

            sub_matrix = vstack(matrix_parts)

        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        print(f"  matrix.shape: {matrix.shape}")

        new_symbol = pybamm.Matrix(matrix) @ discretised_symbol + bcs_vector
        print(f"  new_symbol.shape: {new_symbol.shape}")

        return new_symbol, domain

    def add_neumann_values(self, symbol, discretised_gradient, bcs, domain):
        """
        Add the known values of the gradient from Neumann boundary conditions to
        the discretised gradient.

        Dirichlet bcs are implemented using ghost nodes, see
        :meth:`pybamm.FiniteVolume3D.add_ghost_nodes`.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_gradient : :class:`pybamm.Vector`
            Contains the discretised gradient of symbol
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left", "right", "top", "bottom", "front", "back") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")
        domain : list of strings
            The domain of the gradient of the symbol (may include ghost nodes)

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_gradient + bcs_vector`. When evaluated, this gives the
            discretised_gradient, with the values of the Neumann boundary conditions
            concatenated at each end (if given).

        """
        submesh = self.mesh[domain]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        lbc_value, lbc_type = bcs.get("left", (None, None))
        rbc_value, rbc_type = bcs.get("right", (None, None))
        fbc_value, fbc_type = bcs.get("front", (None, None))
        bbc_value, bbc_type = bcs.get("back", (None, None))
        botbc_value, botbc_type = bcs.get("bottom", (None, None))
        tbc_value, tbc_type = bcs.get("top", (None, None))

        print(f"Adding Neumann boundary conditions to {symbol} in domain {domain}")

        if lbc_type == "Neumann" or rbc_type == "Neumann":
            # X-direction gradient processing - only count X boundary conditions
            n_bcs = 0
            if lbc_type == "Neumann":
                n_bcs += 1
            if rbc_type == "Neumann":
                n_bcs += 1
            print(
                f"Processing X-direction gradient with {n_bcs} X-boundary Neumann conditions"
            )

            # Create all vectors with X-gradient sizing
            base_size = (n_x - 1 + n_bcs) * second_dim_repeats * (n_y * n_z)

            # X-direction boundaries (active)
            if lbc_type == "Neumann" and lbc_value != 0:
                lbc_sub_matrix = coo_matrix(
                    ([1], ([0], [0])), shape=(n_x - 1 + n_bcs, 1)
                )
                lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
                lbc_matrix = vstack([lbc_matrix] * (n_y * n_z))
                if lbc_value.evaluates_to_number():
                    left_bc = lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    left_bc = lbc_value
                lbc_vector = pybamm.Matrix(lbc_matrix) @ left_bc
            else:
                lbc_vector = pybamm.Vector(np.zeros(base_size))

            if rbc_type == "Neumann" and rbc_value != 0:
                rbc_sub_matrix = coo_matrix(
                    ([1], ([n_x + n_bcs - 2], [0])), shape=(n_x - 1 + n_bcs, 1)
                )
                rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
                rbc_matrix = vstack([rbc_matrix] * (n_y * n_z))
                if rbc_value.evaluates_to_number():
                    right_bc = rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    right_bc = rbc_value
                rbc_vector = pybamm.Matrix(rbc_matrix) @ right_bc
            else:
                rbc_vector = pybamm.Vector(np.zeros(base_size))

            # Y and Z direction boundaries (inactive for X-gradient)
            fbc_vector = pybamm.Vector(np.zeros(base_size))
            bbc_vector = pybamm.Vector(np.zeros(base_size))
            botbc_vector = pybamm.Vector(np.zeros(base_size))
            tbc_vector = pybamm.Vector(np.zeros(base_size))

            # X-direction matrix construction
            left_vector = csr_matrix((1, n_x - 1)) if lbc_type == "Neumann" else None
            right_vector = csr_matrix((1, n_x - 1)) if rbc_type == "Neumann" else None
            sub_matrix = vstack(
                [m for m in [left_vector, eye(n_x - 1), right_vector] if m is not None]
            )
            sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif fbc_type == "Neumann" or bbc_type == "Neumann":
            # Y-direction gradient processing - only count Y boundary conditions
            n_bcs = 0
            if fbc_type == "Neumann":
                n_bcs += 1
            if bbc_type == "Neumann":
                n_bcs += 1
            print(
                f"Processing Y-direction gradient with {n_bcs} Y-boundary Neumann conditions"
            )

            # Create all vectors with Y-gradient sizing
            base_size = (n_y - 1 + n_bcs) * second_dim_repeats * (n_x * n_z)

            # Y-direction boundaries (active)
            if fbc_type == "Neumann" and fbc_value != 0:
                row_indices = np.arange(0, n_x * n_z)
                col_indices = np.zeros(len(row_indices))
                vals = np.ones(len(row_indices))
                fbc_sub_matrix = coo_matrix(
                    (vals, (row_indices, col_indices)),
                    shape=((n_y - 1 + n_bcs) * n_x * n_z, 1),
                )
                fbc_matrix = csr_matrix(kron(eye(second_dim_repeats), fbc_sub_matrix))
                if fbc_value.evaluates_to_number():
                    front_bc = fbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    front_bc = fbc_value
                fbc_vector = pybamm.Matrix(fbc_matrix) @ front_bc
            else:
                fbc_vector = pybamm.Vector(np.zeros(base_size))

            if bbc_type == "Neumann" and bbc_value != 0:
                row_indices = np.arange(
                    (n_x * n_z * (n_y - 1 + n_bcs)) - n_x * n_z,
                    n_x * n_z * (n_y - 1 + n_bcs),
                )
                col_indices = np.zeros(len(row_indices))
                vals = np.ones(len(row_indices))
                bbc_sub_matrix = coo_matrix(
                    (vals, (row_indices, col_indices)),
                    shape=((n_y - 1 + n_bcs) * n_x * n_z, 1),
                )
                bbc_matrix = csr_matrix(kron(eye(second_dim_repeats), bbc_sub_matrix))
                if bbc_value.evaluates_to_number():
                    back_bc = bbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    back_bc = bbc_value
                bbc_vector = pybamm.Matrix(bbc_matrix) @ back_bc
            else:
                bbc_vector = pybamm.Vector(np.zeros(base_size))

            # X and Z direction boundaries (inactive for Y-gradient)
            lbc_vector = pybamm.Vector(np.zeros(base_size))
            rbc_vector = pybamm.Vector(np.zeros(base_size))
            botbc_vector = pybamm.Vector(np.zeros(base_size))
            tbc_vector = pybamm.Vector(np.zeros(base_size))

            # Y-direction matrix construction
            front_vector = (
                csr_matrix((n_x * n_z, (n_y - 1) * n_x * n_z))
                if fbc_type == "Neumann"
                else None
            )
            back_vector = (
                csr_matrix((n_x * n_z, (n_y - 1) * n_x * n_z))
                if bbc_type == "Neumann"
                else None
            )
            sub_matrix = vstack(
                [
                    m
                    for m in [front_vector, eye((n_y - 1) * n_x * n_z), back_vector]
                    if m is not None
                ]
            )

        elif botbc_type == "Neumann" or tbc_type == "Neumann":
            # Z-direction gradient processing - only count Z boundary conditions
            n_bcs = 0
            if botbc_type == "Neumann":
                n_bcs += 1
            if tbc_type == "Neumann":
                n_bcs += 1
            print(
                f"Processing Z-direction gradient with {n_bcs} Z-boundary Neumann conditions"
            )

            # Create all vectors with Z-gradient sizing
            base_size = (n_z - 1 + n_bcs) * second_dim_repeats * (n_x * n_y)

            # Z-direction boundaries (active)
            if botbc_type == "Neumann" and botbc_value != 0:
                row_indices = np.arange(0, n_x * n_y)
                col_indices = np.zeros(len(row_indices))
                vals = np.ones(len(row_indices))
                botbc_sub_matrix = coo_matrix(
                    (vals, (row_indices, col_indices)),
                    shape=((n_z - 1 + n_bcs) * n_x * n_y, 1),
                )
                botbc_matrix = csr_matrix(
                    kron(eye(second_dim_repeats), botbc_sub_matrix)
                )
                if botbc_value.evaluates_to_number():
                    bottom_bc = botbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    bottom_bc = botbc_value
                botbc_vector = pybamm.Matrix(botbc_matrix) @ bottom_bc
            else:
                botbc_vector = pybamm.Vector(np.zeros(base_size))

            if tbc_type == "Neumann" and tbc_value != 0:
                row_indices = np.arange(
                    (n_x * n_y * (n_z - 1 + n_bcs)) - n_x * n_y,
                    n_x * n_y * (n_z - 1 + n_bcs),
                )
                col_indices = np.zeros(len(row_indices))
                vals = np.ones(len(row_indices))
                tbc_sub_matrix = coo_matrix(
                    (vals, (row_indices, col_indices)),
                    shape=((n_z - 1 + n_bcs) * n_x * n_y, 1),
                )
                tbc_matrix = csr_matrix(kron(eye(second_dim_repeats), tbc_sub_matrix))
                if tbc_value.evaluates_to_number():
                    top_bc = tbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                else:
                    top_bc = tbc_value
                tbc_vector = pybamm.Matrix(tbc_matrix) @ top_bc
            else:
                tbc_vector = pybamm.Vector(np.zeros(base_size))

            # X and Y direction boundaries (inactive for Z-gradient)
            lbc_vector = pybamm.Vector(np.zeros(base_size))
            rbc_vector = pybamm.Vector(np.zeros(base_size))
            fbc_vector = pybamm.Vector(np.zeros(base_size))
            bbc_vector = pybamm.Vector(np.zeros(base_size))

            # Z-direction matrix construction
            bottom_vector = (
                csr_matrix((n_x * n_y, (n_z - 1) * n_x * n_y))
                if botbc_type == "Neumann"
                else None
            )
            top_vector = (
                csr_matrix((n_x * n_y, (n_z - 1) * n_x * n_y))
                if tbc_type == "Neumann"
                else None
            )
            sub_matrix = vstack(
                [
                    m
                    for m in [bottom_vector, eye((n_z - 1) * n_x * n_y), top_vector]
                    if m is not None
                ]
            )

        else:
            return discretised_gradient

        print(
            f" {lbc_vector.shape}, {rbc_vector.shape}, {fbc_vector.shape}, "
            f"{bbc_vector.shape}, {botbc_vector.shape}, {tbc_vector.shape}"
        )

        bcs_vector = (
            lbc_vector
            + rbc_vector
            + fbc_vector
            + bbc_vector
            + botbc_vector
            + tbc_vector
        )

        print(f"  bcs_vector.shape: {bcs_vector.shape}")
        bcs_vector.copy_domains(discretised_gradient)

        print(f"  sub_matrix.shape: {sub_matrix.shape}")
        # repeat matrix for secondary dimensions
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        print(f"  matrix.shape: {matrix.shape}")

        new_gradient = pybamm.Matrix(matrix) @ discretised_gradient + bcs_vector

        return new_gradient

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Uses extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method for 3D meshes.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """
        # Find the submesh
        submesh = self.mesh[discretised_child.domain]

        if "-" in symbol.side:
            sides = symbol.side.split("-")
            primary_side = sides[0]
            secondary_side = sides[1] if len(sides) > 1 else None
            tertiary_side = sides[2] if len(sides) > 2 else None
        else:
            primary_side = symbol.side
            secondary_side = None
            tertiary_side = None

        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        if bcs is None:
            bcs = {}

        extrap_order_gradient = self.options["extrapolation"]["order"]["gradient"]
        extrap_order_value = self.options["extrapolation"]["order"]["value"]
        use_bcs = self.options["extrapolation"]["use bcs"]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        nodes_x = submesh.nodes_x
        edges_x = submesh.edges_x
        nodes_y = submesh.nodes_y
        edges_y = submesh.edges_y
        nodes_z = submesh.nodes_z
        edges_z = submesh.edges_z

        dx0_x = nodes_x[0] - edges_x[0]
        dx1_x = submesh.d_nodes_x[0]
        dxN_x = edges_x[-1] - nodes_x[-1]
        dxNm1_x = submesh.d_nodes_x[-1]

        dx0_y = nodes_y[0] - edges_y[0]
        dx1_y = submesh.d_nodes_y[0]
        dxN_y = edges_y[-1] - nodes_y[-1]
        dxNm1_y = submesh.d_nodes_y[-1]

        dx0_z = nodes_z[0] - edges_z[0]
        dx1_z = submesh.d_nodes_z[0]
        dxN_z = edges_z[-1] - nodes_z[-1]
        dxNm1_z = submesh.d_nodes_z[-1]

        child = symbol.child

        # Initialize variables
        sub_matrix = None
        additive = pybamm.Scalar(0)

        if isinstance(symbol, pybamm.BoundaryValue):
            if use_bcs and pybamm.has_bc_of_form(child, primary_side, bcs, "Dirichlet"):
                if primary_side in ["left", "right"]:
                    sub_matrix = csr_matrix((n_y * n_z, n_x * n_y * n_z))
                elif primary_side in ["front", "back"]:
                    sub_matrix = csr_matrix((n_x * n_z, n_x * n_y * n_z))
                elif primary_side in ["bottom", "top"]:
                    sub_matrix = csr_matrix((n_x * n_y, n_x * n_y * n_z))
                additive = bcs[child][primary_side][0]

            elif primary_side == "left":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices = np.arange(0, n_x * n_y * n_z, n_x)
                        vals = np.ones(n_y * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_x
                        dx1 = dx1_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                        col_indices_1 = col_indices_0 + 1
                        vals_0 = np.ones(n_y * n_z) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_y * n_z) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for left boundary"
                    )

            elif primary_side == "right":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices = np.arange(n_x - 1, n_x * n_y * n_z, n_x)
                        vals = np.ones(n_y * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_x
                        dxNm1 = dxNm1_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                        col_indices_N = col_indices_Nm1 + 1
                        vals_Nm1 = np.ones(n_y * n_z) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_y * n_z) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for right boundary"
                    )

            elif primary_side == "front":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices.append(k * n_x * n_y + i)
                        col_indices = np.array(col_indices)
                        vals = np.ones(n_x * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_y
                        dx1 = dx1_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices_0 = []
                        col_indices_1 = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices_0.append(k * n_x * n_y + i)
                                col_indices_1.append(k * n_x * n_y + n_x + i)
                        col_indices_0 = np.array(col_indices_0)
                        col_indices_1 = np.array(col_indices_1)
                        vals_0 = np.ones(n_x * n_z) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_x * n_z) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for front boundary"
                    )

            elif primary_side == "back":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                        col_indices = np.array(col_indices)
                        vals = np.ones(n_x * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_y
                        dxNm1 = dxNm1_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices_Nm1 = []
                        col_indices_N = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices_Nm1.append(
                                    k * n_x * n_y + (n_y - 2) * n_x + i
                                )
                                col_indices_N.append(
                                    k * n_x * n_y + (n_y - 1) * n_x + i
                                )
                        col_indices_Nm1 = np.array(col_indices_Nm1)
                        col_indices_N = np.array(col_indices_N)
                        vals_Nm1 = np.ones(n_x * n_z) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_x * n_z) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for back boundary"
                    )

            elif primary_side == "bottom":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices = np.arange(0, n_x * n_y)
                        vals = np.ones(n_x * n_y)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_z
                        dx1 = dx1_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices_0 = np.arange(0, n_x * n_y)
                        col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                        vals_0 = np.ones(n_x * n_y) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_x * n_y) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for bottom boundary"
                    )

            elif primary_side == "top":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                        vals = np.ones(n_x * n_y)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_z
                        dxNm1 = dxNm1_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices_Nm1 = np.arange(
                            (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                        )
                        col_indices_N = np.arange(
                            (n_z - 1) * n_x * n_y, n_z * n_x * n_y
                        )
                        vals_Nm1 = np.ones(n_x * n_y) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_x * n_y) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for top boundary"
                    )

            else:
                raise ValueError(f"Unknown primary side: {primary_side}")

            # Handle secondary sides (compound boundaries like edges)
            if secondary_side is not None:
                additive_secondary = pybamm.Scalar(0)

                if secondary_side == "front":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_y
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        tertiary_matrix = csr_matrix(([1], ([0], [0])), shape=(1, 1))
                    else:
                        dx0 = dx0_y
                        dx1 = dx1_y
                        tertiary_matrix = csr_matrix(
                            ([1 + (dx0 / dx1)], ([0], [0])), shape=(1, 1)
                        )

                    sub_matrix = tertiary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "back":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_y
                        additive_secondary = dxN * bcs[child][secondary_side][0]
                        tertiary_matrix = csr_matrix(([1], ([0], [0])), shape=(1, 1))
                    else:
                        dxN = dxN_y
                        dxNm1 = dxNm1_y
                        tertiary_matrix = csr_matrix(
                            ([1 + (dxN / dxNm1)], ([0], [0])), shape=(1, 1)
                        )

                    sub_matrix = tertiary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "bottom":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_z
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        tertiary_matrix = csr_matrix(([1], ([0], [0])), shape=(1, 1))
                    else:
                        dx0 = dx0_z
                        dx1 = dx1_z
                        tertiary_matrix = csr_matrix(
                            ([1 + (dx0 / dx1)], ([0], [0])), shape=(1, 1)
                        )

                    sub_matrix = tertiary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "top":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_z
                        additive_secondary = dxN * bcs[child][secondary_side][0]
                        tertiary_matrix = csr_matrix(([1], ([0], [0])), shape=(1, 1))
                    else:
                        dxN = dxN_z
                        dxNm1 = dxNm1_z
                        tertiary_matrix = csr_matrix(
                            ([1 + (dxN / dxNm1)], ([0], [0])), shape=(1, 1)
                        )

                    sub_matrix = tertiary_matrix @ sub_matrix
                    additive += additive_secondary

                # Handle tertiary sides (corner boundaries)
                if tertiary_side is not None:
                    additive_tertiary = pybamm.Scalar(0)

                    if tertiary_side == "bottom":
                        if use_bcs and pybamm.has_bc_of_form(
                            child, tertiary_side, bcs, "Neumann"
                        ):
                            dx0 = dx0_z
                            additive_tertiary = -dx0 * bcs[child][tertiary_side][0]
                            tertiary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, 1)
                            )
                        else:
                            dx0 = dx0_z
                            dx1 = dx1_z
                            tertiary_matrix = csr_matrix(
                                ([1 + (dx0 / dx1)], ([0], [0])), shape=(1, 1)
                            )

                        sub_matrix = tertiary_matrix @ sub_matrix
                        additive += additive_tertiary

                    elif tertiary_side == "top":
                        if use_bcs and pybamm.has_bc_of_form(
                            child, tertiary_side, bcs, "Neumann"
                        ):
                            dxN = dxN_z
                            additive_tertiary = dxN * bcs[child][tertiary_side][0]
                            tertiary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, 1)
                            )
                        else:
                            dxN = dxN_z
                            dxNm1 = dxNm1_z
                            tertiary_matrix = csr_matrix(
                                ([1 + (dxN / dxNm1)], ([0], [0])), shape=(1, 1)
                            )

                        sub_matrix = tertiary_matrix @ sub_matrix
                        additive += additive_tertiary

        elif isinstance(symbol, pybamm.BoundaryGradient):
            if use_bcs and pybamm.has_bc_of_form(child, primary_side, bcs, "Neumann"):
                if primary_side in ["left", "right"]:
                    sub_matrix = csr_matrix((n_y * n_z, n_x * n_y * n_z))
                elif primary_side in ["front", "back"]:
                    sub_matrix = csr_matrix((n_x * n_z, n_x * n_y * n_z))
                elif primary_side in ["bottom", "top"]:
                    sub_matrix = csr_matrix((n_x * n_y, n_x * n_y * n_z))
                additive = bcs[child][primary_side][0]

            elif primary_side == "left":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                    col_indices_1 = col_indices_0 + 1
                    vals_0 = np.ones(n_y * n_z) * (-1 / dx1)
                    vals_1 = np.ones(n_y * n_z) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    # FIX: Implement quadratic extrapolation
                    dx1 = dx1_x
                    dx2 = submesh.d_nodes_x[1] if len(submesh.d_nodes_x) > 1 else dx1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                    col_indices_1 = col_indices_0 + 1
                    col_indices_2 = col_indices_0 + 2 if n_x > 2 else col_indices_0 + 1

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_y * n_z) * c0
                    vals_1 = np.ones(n_y * n_z) * c1
                    vals_2 = np.ones(n_y * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)

                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for left gradient"
                    )

            elif primary_side == "right":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    col_indices_N = col_indices_Nm1 + 1
                    vals_Nm1 = np.ones(n_y * n_z) * (-1 / dxNm1)
                    vals_N = np.ones(n_y * n_z) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_x
                    dxNm2 = (
                        submesh.d_nodes_x[-2] if len(submesh.d_nodes_x) > 1 else dxNm1_x
                    )
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_Nm2 = (
                        np.arange(n_x - 3, n_x * n_y * n_z, n_x)
                        if n_x > 2
                        else np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    )
                    col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    col_indices_N = col_indices_Nm1 + 1

                    # Quadratic extrapolation coefficients for right boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_y * n_z) * c0
                    vals_Nm1 = np.ones(n_y * n_z) * c1
                    vals_N = np.ones(n_y * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for right gradient"
                    )

            elif primary_side == "front":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_0 = []
                    col_indices_1 = []
                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_0.append(k * n_x * n_y + i)
                            col_indices_1.append(k * n_x * n_y + n_x + i)
                    col_indices_0 = np.array(col_indices_0)
                    col_indices_1 = np.array(col_indices_1)
                    vals_0 = np.ones(n_x * n_z) * (-1 / dx1)
                    vals_1 = np.ones(n_x * n_z) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dx1 = dx1_y
                    dx2 = submesh.d_nodes_y[1] if len(submesh.d_nodes_y) > 1 else dx1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_0 = []
                    col_indices_1 = []
                    col_indices_2 = []

                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_0.append(k * n_x * n_y + i)
                            col_indices_1.append(k * n_x * n_y + n_x + i)
                            col_indices_2.append(
                                k * n_x * n_y + 2 * n_x + i
                                if n_y > 2
                                else k * n_x * n_y + n_x + i
                            )

                    col_indices_0 = np.array(col_indices_0)
                    col_indices_1 = np.array(col_indices_1)
                    col_indices_2 = np.array(col_indices_2)

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_x * n_z) * c0
                    vals_1 = np.ones(n_x * n_z) * c1
                    vals_2 = np.ones(n_x * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for front gradient"
                    )

            elif primary_side == "back":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_Nm1 = []
                    col_indices_N = []
                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_Nm1.append(k * n_x * n_y + (n_y - 2) * n_x + i)
                            col_indices_N.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                    col_indices_Nm1 = np.array(col_indices_Nm1)
                    col_indices_N = np.array(col_indices_N)
                    vals_Nm1 = np.ones(n_x * n_z) * (-1 / dxNm1)
                    vals_N = np.ones(n_x * n_z) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_y
                    dxNm2 = (
                        submesh.d_nodes_y[-2] if len(submesh.d_nodes_y) > 1 else dxNm1_y
                    )
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_Nm2 = []
                    col_indices_Nm1 = []
                    col_indices_N = []

                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_Nm2.append(
                                k * n_x * n_y + (n_y - 3) * n_x + i
                                if n_y > 2
                                else k * n_x * n_y + (n_y - 2) * n_x + i
                            )
                            col_indices_Nm1.append(k * n_x * n_y + (n_y - 2) * n_x + i)
                            col_indices_N.append(k * n_x * n_y + (n_y - 1) * n_x + i)

                    col_indices_Nm2 = np.array(col_indices_Nm2)
                    col_indices_Nm1 = np.array(col_indices_Nm1)
                    col_indices_N = np.array(col_indices_N)

                    # Quadratic extrapolation coefficients for back boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_x * n_z) * c0
                    vals_Nm1 = np.ones(n_x * n_z) * c1
                    vals_N = np.ones(n_x * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for back gradient"
                    )

            elif primary_side == "bottom":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_0 = np.arange(0, n_x * n_y)
                    col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                    vals_0 = np.ones(n_x * n_y) * (-1 / dx1)
                    vals_1 = np.ones(n_x * n_y) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dx1 = dx1_z
                    dx2 = submesh.d_nodes_z[1] if len(submesh.d_nodes_z) > 1 else dx1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_0 = np.arange(0, n_x * n_y)
                    col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                    col_indices_2 = (
                        np.arange(2 * n_x * n_y, 3 * n_x * n_y)
                        if n_z > 2
                        else np.arange(n_x * n_y, 2 * n_x * n_y)
                    )

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_x * n_y) * c0
                    vals_1 = np.ones(n_x * n_y) * c1
                    vals_2 = np.ones(n_x * n_y) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for bottom gradient"
                    )

            elif primary_side == "top":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_Nm1 = np.arange(
                        (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                    )
                    col_indices_N = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                    vals_Nm1 = np.ones(n_x * n_y) * (-1 / dxNm1)
                    vals_N = np.ones(n_x * n_y) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_z
                    dxNm2 = (
                        submesh.d_nodes_z[-2] if len(submesh.d_nodes_z) > 1 else dxNm1_z
                    )
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_Nm2 = (
                        np.arange((n_z - 3) * n_x * n_y, (n_z - 2) * n_x * n_y)
                        if n_z > 2
                        else np.arange((n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y)
                    )
                    col_indices_Nm1 = np.arange(
                        (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                    )
                    col_indices_N = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)

                    # Quadratic extrapolation coefficients for top boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_x * n_y) * c0
                    vals_Nm1 = np.ones(n_x * n_y) * c1
                    vals_N = np.ones(n_x * n_y) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for top gradient"
                    )

            else:
                raise NotImplementedError(
                    f"Gradient boundary condition for {primary_side} not implemented"
                )

        else:
            raise ValueError(f"Unknown symbol type: {type(symbol)}")

        # Fallback if sub_matrix is still None or empty
        if sub_matrix is None or sub_matrix.nnz == 0:
            # Create simple extraction matrix based on primary_side
            if primary_side in ["left", "right"]:
                shape = (n_y * n_z, n_x * n_y * n_z)
                rows = np.arange(n_y * n_z)
                if primary_side == "left":
                    cols = np.arange(0, n_x * n_y * n_z, n_x)
                else:  # right
                    cols = np.arange(n_x - 1, n_x * n_y * n_z, n_x)

            elif primary_side in ["front", "back"]:
                shape = (n_x * n_z, n_x * n_y * n_z)
                rows = np.arange(n_x * n_z)
                cols = []
                for k in range(n_z):
                    for i in range(n_x):
                        if primary_side == "front":
                            cols.append(k * n_x * n_y + i)
                        else:  # back
                            cols.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                cols = np.array(cols)

            else:  # bottom, top
                shape = (n_x * n_y, n_x * n_y * n_z)
                rows = np.arange(n_x * n_y)
                if primary_side == "bottom":
                    cols = np.arange(n_x * n_y)
                else:  # top
                    cols = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)

            vals = np.ones(len(rows))
            sub_matrix = csr_matrix((vals, (rows, cols)), shape=shape)

        # Create final matrix
        matrix = csr_matrix(kron(eye(repeats), sub_matrix))
        matrix = pybamm.Matrix(matrix)
        boundary_value = matrix @ discretised_child
        boundary_value.copy_domains(symbol)

        additive.copy_domains(symbol)
        boundary_value += additive

        return boundary_value

    def evaluate_at(self, symbol, discretised_child, position):
        """
        Evaluate a symbol at a specific position in 3D space.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The boundary value or flux symbol.
        discretised_child : :class:`pybamm.StateVector`
            The discretised variable from which to calculate the value.
        position : list or array
            The point in 3D space at which to evaluate the symbol.

        Returns
        -------
        :class:`pybamm.MatrixMultiplication`
            The variable representing the value at the given point.
        """
        submesh = self.mesh[symbol.domain]
        x_pos, y_pos, z_pos = position

        # Find closest indices
        x_idx = np.argmin(np.abs(submesh.nodes_x - x_pos))
        y_idx = np.argmin(np.abs(submesh.nodes_y - y_pos))
        z_idx = np.argmin(np.abs(submesh.nodes_z - z_pos))

        # Convert to linear index
        linear_idx = (
            z_idx * submesh.npts_x * submesh.npts_y + y_idx * submesh.npts_x + x_idx
        )

        # Create selection matrix
        selection_matrix = csr_matrix(
            ([1], ([0], [linear_idx])), shape=(1, submesh.npts)
        )

        return pybamm.Matrix(selection_matrix) @ discretised_child

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Discretise binary operators in model equations for 3D finite volume.
        Performs appropriate averaging of diffusivities if one of the children is a
        gradient operator, so that discretised sizes match up. For this averaging we
        use the harmonic mean [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        bin_op : :class:`pybamm.BinaryOperator`
            Binary operator to discretise
        left : :class:`pybamm.Symbol`
            The left child of `bin_op`
        right : :class:`pybamm.Symbol`
            The right child of `bin_op`
        disc_left : :class:`pybamm.Symbol`
            The discretised left child of `bin_op`
        disc_right : :class:`pybamm.Symbol`
            The discretised right child of `bin_op`
        Returns
        -------
        :class:`pybamm.BinaryOperator`
            Discretised binary operator
        """

        # Post-processing to make sure discretised dimensions match
        left_evaluates_on_edges = left.evaluates_on_edges("primary")
        right_evaluates_on_edges = right.evaluates_on_edges("primary")

        # Handle 3D VectorField cases
        if hasattr(disc_left, "x_field") and hasattr(disc_right, "x_field"):
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_left_x = self.node_to_edge(
                    disc_left.x_field, method=method, direction="x"
                )
                disc_left_y = self.node_to_edge(
                    disc_left.y_field, method=method, direction="y"
                )
                disc_left_z = self.node_to_edge(
                    disc_left.z_field, method=method, direction="z"
                )
                disc_left = pybamm.VectorField3D(disc_left_x, disc_left_y, disc_left_z)

            elif left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_right_x = self.node_to_edge(
                    disc_right.x_field, method=method, direction="x"
                )
                disc_right_y = self.node_to_edge(
                    disc_right.y_field, method=method, direction="y"
                )
                disc_right_z = self.node_to_edge(
                    disc_right.z_field, method=method, direction="z"
                )
                disc_right = pybamm.VectorField3D(
                    disc_right_x, disc_right_y, disc_right_z
                )

            x_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.x_field, disc_right])
            )
            y_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.y_field, disc_right])
            )
            z_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.z_field, disc_right])
            )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        elif hasattr(disc_left, "x_field") and not hasattr(disc_right, "x_field"):
            if left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_right_x = self.node_to_edge(
                    disc_right, method=method, direction="x"
                )
                disc_right_y = self.node_to_edge(
                    disc_right, method=method, direction="y"
                )
                disc_right_z = self.node_to_edge(
                    disc_right, method=method, direction="z"
                )
                disc_right_vector = pybamm.VectorField3D(
                    disc_right_x, disc_right_y, disc_right_z
                )

                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.x_field, disc_right_vector.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.y_field, disc_right_vector.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.z_field, disc_right_vector.z_field])
                )
            else:
                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.x_field, disc_right])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.y_field, disc_right])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.z_field, disc_right])
                )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        elif not hasattr(disc_left, "x_field") and hasattr(disc_right, "x_field"):
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"
                disc_left_x = self.node_to_edge(disc_left, method=method, direction="x")
                disc_left_y = self.node_to_edge(disc_left, method=method, direction="y")
                disc_left_z = self.node_to_edge(disc_left, method=method, direction="z")
                disc_left_vector = pybamm.VectorField3D(
                    disc_left_x, disc_left_y, disc_left_z
                )

                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.x_field, disc_right.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.y_field, disc_right.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.z_field, disc_right.z_field])
                )
            else:
                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.z_field])
                )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        if isinstance(bin_op, pybamm.Inner):
            if left.evaluates_on_edges("primary"):
                # Check if disc_left is already node-based (e.g. from VectorField3D component handling)
                if not (
                    hasattr(disc_left, "domains")
                    and self.mesh[disc_left.domain].npts == disc_left.size
                ):  # A heuristic
                    disc_left = self.edge_to_node(disc_left)
            if right.evaluates_on_edges("primary"):
                if not (
                    hasattr(disc_right, "domains")
                    and self.mesh[disc_right.domain].npts == disc_right.size
                ):  # A heuristic
                    disc_right = self.edge_to_node(disc_right)

        # If neither child evaluates on edges, or both children have gradients,
        # no need to do any averaging
        elif left_evaluates_on_edges == right_evaluates_on_edges:
            pass
        # If only left child evaluates on edges, map right child onto edges
        # using the harmonic mean if the left child is a gradient (i.e. this
        # binary operator represents a flux)
        elif left_evaluates_on_edges and not right_evaluates_on_edges:
            if isinstance(left, pybamm.Gradient):
                method = "harmonic"
            else:
                method = "arithmetic"
            disc_right = self.node_to_edge(disc_right, method=method)

        # If only right child evaluates on edges, map left child onto edges
        # using the harmonic mean if the right child is a gradient (i.e. this
        # binary operator represents a flux)
        elif right_evaluates_on_edges and not left_evaluates_on_edges:
            if isinstance(right, pybamm.Gradient):
                method = "harmonic"
            else:
                method = "arithmetic"
            disc_left = self.node_to_edge(disc_left, method=method)

        # Return new binary operator with appropriate class
        out = pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))

        return out

    def concatenation(self, disc_children):
        """
        Discrete concatenation for 3D finite-volume.

        Any child vector of length n_edges (i.e. defined on edges) is
        first averaged to nodes via self.edge_to_node(). All others
        must then be node-valued (length n_nodes). Finally we
        call domain_concatenation() which handles the x→y→z flattening.
        """
        for idx, child in enumerate(disc_children):
            submesh = self.mesh[child.domain]
            repeats = self._get_auxiliary_domain_repeats(child.domains)

            # total number of cell centers (nodes) in 3D = npts_x * npts_y * npts_z
            n_nodes = submesh.npts_x * submesh.npts_y * submesh.npts_z * repeats

            # total number of edges: sum over x , y and z directed edges
            n_ex = (submesh.npts_x - 1) * submesh.npts_y * submesh.npts_z
            n_ey = submesh.npts_x * (submesh.npts_y - 1) * submesh.npts_z
            n_ez = submesh.npts_x * submesh.npts_y * (submesh.npts_z - 1)
            n_edges = (n_ex + n_ey + n_ez) * repeats

            if child.size == n_edges:
                # average and replace in place
                disc_children[idx] = self.edge_to_node(child)
            elif child.size != n_nodes:
                raise pybamm.ShapeError(
                    f"In 3D concatenation expected child size to be n_nodes={n_nodes} "
                    f"or n_edges={n_edges}, but got {child.size}"
                )

        return pybamm.domain_concatenation(disc_children, self.mesh)

    def edge_to_node(self, discretised_symbol, method="arithmetic", direction="x"):
        """
        Convert a discretised symbol evaluated on the cell edges to a discretised symbol
        evaluated on the cell nodes.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "edge to node", method, direction)

    def node_to_edge(self, discretised_symbol, method="arithmetic", direction="x"):
        """
        Convert a discretised symbol evaluated on the cell nodes to a discretised symbol
        evaluated on the cell edges.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "node to edge", method, direction)

    def shift(self, discretised_symbol, shift_key, method, direction="x"):
        """
        Convert a discretised symbol evaluated at edges/nodes, to a discretised symbol
        evaluated at nodes/edges. Can be the arithmetic mean or the harmonic mean.

        Note: when computing fluxes at cell edges it is better to take the
        harmonic mean based on [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.Symbol`
            Symbol to be averaged. When evaluated, this symbol returns either a scalar
            or an array of shape (n,) or (n+1,), where n is the number of points in the
            mesh for the symbol's domain (n = self.mesh[symbol.domain].npts)
        shift_key : str
            Whether to shift from nodes to edges ("node to edge"), or from edges to
            nodes ("edge to node")
        method : str
            Whether to use the "arithmetic" or "harmonic" mean
        direction : str
            Direction along which to shift ("x", "y", or "z")

        Returns
        -------
        :class:`pybamm.Symbol`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n+1,) (if `shift_key = "node to edge"`) or (n,) (if
            `shift_key = "edge to node"`)
        """

        def arithmetic_mean(array, direction):
            """Calculate the arithmetic mean of an array using matrix multiplication"""
            submesh = self.mesh[array.domain]
            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

            if shift_key == "node to edge":
                if direction == "x":
                    sub_matrix_left = csr_matrix(
                        ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n_x)
                    )
                    sub_matrix_center = diags([0.5, 0.5], [0, 1], shape=(n_x - 1, n_x))
                    sub_matrix_right = csr_matrix(
                        ([-0.5, 1.5], ([0, 0], [n_x - 2, n_x - 1])), shape=(1, n_x)
                    )
                    sub_matrix_1d = vstack(
                        [sub_matrix_left, sub_matrix_center, sub_matrix_right]
                    )
                    sub_matrix = block_diag((sub_matrix_1d,) * (n_y * n_z))

                elif direction == "y":
                    one_fives_front = np.ones(n_x) * 1.5
                    neg_zero_fives_front = np.ones(n_x) * -0.5
                    rows = np.arange(0, n_x)
                    cols_first = np.arange(0, n_x)
                    cols_second = np.arange(n_x, 2 * n_x)
                    data = np.hstack([one_fives_front, neg_zero_fives_front])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    sub_matrix_front = csr_matrix(
                        (data, (rows, cols)), shape=(n_x, n_x * n_y)
                    )

                    cols_first = np.arange((n_y - 2) * n_x, (n_y - 1) * n_x)
                    cols_second = np.arange((n_y - 1) * n_x, n_y * n_x)
                    data = np.hstack([neg_zero_fives_front, one_fives_front])
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_back = csr_matrix(
                        (data, (rows, cols)), shape=(n_x, n_x * n_y)
                    )

                    data = np.ones((n_y - 1) * n_x) * 0.5
                    data = np.hstack([data, data])
                    rows = np.arange(0, (n_y - 1) * n_x)
                    rows = np.hstack([rows, rows])
                    cols_first = np.arange(0, (n_y - 1) * n_x)
                    cols_second = np.arange(n_x, n_x * n_y)
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_center = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * (n_y - 1), n_x * n_y)
                    )
                    sub_matrix_2d = vstack(
                        [
                            sub_matrix_front,
                            sub_matrix_center,
                            sub_matrix_back,
                        ]
                    )
                    sub_matrix = block_diag((sub_matrix_2d,) * n_z)

                elif direction == "z":
                    one_fives_bottom = np.ones(n_x * n_y) * 1.5
                    neg_zero_fives_bottom = np.ones(n_x * n_y) * -0.5
                    rows = np.arange(0, n_x * n_y)
                    cols_first = np.arange(0, n_x * n_y)
                    cols_second = np.arange(n_x * n_y, 2 * n_x * n_y)
                    data = np.hstack([one_fives_bottom, neg_zero_fives_bottom])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    sub_matrix_bottom = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * n_y, n_x * n_y * n_z)
                    )

                    cols_first = np.arange((n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y)
                    cols_second = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                    data = np.hstack([neg_zero_fives_bottom, one_fives_bottom])
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_top = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * n_y, n_x * n_y * n_z)
                    )

                    data = np.ones((n_z - 1) * n_x * n_y) * 0.5
                    data = np.hstack([data, data])
                    rows = np.arange(0, (n_z - 1) * n_x * n_y)
                    rows = np.hstack([rows, rows])
                    cols_first = np.arange(0, (n_z - 1) * n_x * n_y)
                    cols_second = np.arange(n_x * n_y, n_x * n_y * n_z)
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_center = csr_matrix(
                        (data, (rows, cols)),
                        shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                    )
                    sub_matrix = vstack(
                        [
                            sub_matrix_bottom,
                            sub_matrix_center,
                            sub_matrix_top,
                        ]
                    )

            elif shift_key == "edge to node":
                if direction == "x":
                    block = diags([0.5, 0.5], [0, 1], shape=(n_x, n_x + 1))
                    sub_matrix = block_diag((block,) * (n_y * n_z))

                elif direction == "y":
                    rows = np.arange(0, n_x * n_y)
                    cols_first = np.arange(0, n_x * n_y)
                    cols_second = np.arange(n_x, n_x * (n_y + 1))
                    data = np.ones(n_x * n_y) * 0.5
                    data = np.hstack([data, data])
                    rows = np.hstack([rows, rows])
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_2d = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * n_y, n_x * (n_y + 1))
                    )
                    sub_matrix = block_diag((sub_matrix_2d,) * n_z)

                elif direction == "z":
                    rows = np.arange(0, n_x * n_y * n_z)
                    cols_first = np.arange(0, n_x * n_y * n_z)
                    cols_second = np.arange(n_x * n_y, n_x * n_y * (n_z + 1))
                    data = np.ones(n_x * n_y * n_z) * 0.5
                    data = np.hstack([data, data])
                    rows = np.hstack([rows, rows])
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix = csr_matrix(
                        (data, (rows, cols)),
                        shape=(n_x * n_y * n_z, n_x * n_y * (n_z + 1)),
                    )

            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")

            # Second dimension repeats
            second_dim_repeats = self._get_auxiliary_domain_repeats(array.domains)
            matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

            return pybamm.Matrix(matrix) @ array

        def harmonic_mean(array, direction):
            """Calculate the harmonic mean following 2D pattern exactly"""
            submesh = self.mesh[array.domain]
            second_dim_repeats = self._get_auxiliary_domain_repeats(
                discretised_symbol.domains
            )

            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

            if shift_key == "node to edge":
                if direction == "x":
                    # Matrix to compute values at the exterior edges
                    edges_sub_matrix_left = csr_matrix(
                        ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n_x)
                    )
                    edges_sub_matrix_center = csr_matrix((n_x - 1, n_x))
                    edges_sub_matrix_right = csr_matrix(
                        ([-0.5, 1.5], ([0, 0], [n_x - 2, n_x - 1])), shape=(1, n_x)
                    )
                    edges_sub_matrix_1d = vstack(
                        [
                            edges_sub_matrix_left,
                            edges_sub_matrix_center,
                            edges_sub_matrix_right,
                        ]
                    )
                    edges_sub_matrix = block_diag((edges_sub_matrix_1d,) * (n_y * n_z))
                    edges_matrix = csr_matrix(
                        kron(eye(second_dim_repeats), edges_sub_matrix)
                    )

                    # Matrix to extract D1 and D2
                    sub_matrix_D1 = hstack([eye(n_x - 1), csr_matrix((n_x - 1, 1))])
                    sub_matrix_D1 = block_diag((sub_matrix_D1,) * (n_y * n_z))
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack([csr_matrix((n_x - 1, 1)), eye(n_x - 1)])
                    sub_matrix_D2 = block_diag((sub_matrix_D2,) * (n_y * n_z))
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute weight beta
                    dx = np.diff(submesh.edges_x)
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_y * n_z, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    # Harmonic mean
                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta))

                    # Padding matrix
                    sub_matrix = vstack(
                        [
                            csr_matrix((1, n_x - 1)),
                            eye(n_x - 1),
                            csr_matrix((1, n_x - 1)),
                        ]
                    )
                    sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
                    matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

                    return (
                        pybamm.Matrix(edges_matrix) @ array
                        + pybamm.Matrix(matrix) @ D_eff
                    )

                elif direction == "y":
                    # Follow 2D tb pattern for y-direction
                    one_fives_front = np.ones(n_x) * 1.5
                    neg_zero_fives_front = np.ones(n_x) * -0.5
                    rows = np.arange(0, n_x)
                    cols_first = np.arange(0, n_x)
                    cols_second = np.arange(n_x, 2 * n_x)
                    data = np.hstack([one_fives_front, neg_zero_fives_front])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    edges_sub_matrix_front = csr_matrix(
                        (data, (rows, cols)), shape=(n_x, n_x * n_y)
                    )

                    cols_first = np.arange((n_y - 2) * n_x, (n_y - 1) * n_x)
                    cols_second = np.arange((n_y - 1) * n_x, n_y * n_x)
                    data = np.hstack([neg_zero_fives_front, one_fives_front])
                    cols = np.hstack([cols_first, cols_second])
                    edges_sub_matrix_back = csr_matrix(
                        (data, (rows, cols)), shape=(n_x, n_x * n_y)
                    )
                    edges_sub_matrix_center = csr_matrix(((n_y - 1) * n_x, n_y * n_x))
                    edges_sub_matrix_2d = vstack(
                        [
                            edges_sub_matrix_front,
                            edges_sub_matrix_center,
                            edges_sub_matrix_back,
                        ]
                    )
                    edges_sub_matrix = block_diag((edges_sub_matrix_2d,) * n_z)
                    edges_matrix = csr_matrix(
                        kron(eye(second_dim_repeats), edges_sub_matrix)
                    )

                    # D1 and D2 matrices
                    sub_matrix_D1 = hstack(
                        [eye(n_x * (n_y - 1)), csr_matrix((n_x * (n_y - 1), n_x))]
                    )
                    sub_matrix_D1 = block_diag((sub_matrix_D1,) * n_z)
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack(
                        [csr_matrix((n_x * (n_y - 1), n_x)), eye(n_x * (n_y - 1))]
                    )
                    sub_matrix_D2 = block_diag((sub_matrix_D2,) * n_z)
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute beta
                    dx = submesh.d_edges_y
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_x, axis=0)
                    sub_beta = np.tile(sub_beta, (n_z, 1))
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta))

                    # Padding matrix
                    sub_matrix_2d = vstack(
                        [
                            csr_matrix((n_x, n_x * (n_y - 1))),
                            eye(n_x * (n_y - 1)),
                            csr_matrix((n_x, n_x * (n_y - 1))),
                        ]
                    )
                    sub_matrix = block_diag((sub_matrix_2d,) * n_z)
                    matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

                    return (
                        pybamm.Matrix(edges_matrix) @ array
                        + pybamm.Matrix(matrix) @ D_eff
                    )

                elif direction == "z":
                    # Follow 2D pattern for z-direction
                    one_fives_bottom = np.ones(n_x * n_y) * 1.5
                    neg_zero_fives_bottom = np.ones(n_x * n_y) * -0.5
                    rows = np.arange(0, n_x * n_y)
                    cols_first = np.arange(0, n_x * n_y)
                    cols_second = np.arange(n_x * n_y, 2 * n_x * n_y)
                    data = np.hstack([one_fives_bottom, neg_zero_fives_bottom])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    edges_sub_matrix_bottom = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * n_y, n_x * n_y * n_z)
                    )

                    cols_first = np.arange((n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y)
                    cols_second = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                    data = np.hstack([neg_zero_fives_bottom, one_fives_bottom])
                    cols = np.hstack([cols_first, cols_second])
                    edges_sub_matrix_top = csr_matrix(
                        (data, (rows, cols)), shape=(n_x * n_y, n_x * n_y * n_z)
                    )
                    edges_sub_matrix_center = csr_matrix(
                        ((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                    )
                    edges_sub_matrix = vstack(
                        [
                            edges_sub_matrix_bottom,
                            edges_sub_matrix_center,
                            edges_sub_matrix_top,
                        ]
                    )
                    edges_matrix = csr_matrix(
                        kron(eye(second_dim_repeats), edges_sub_matrix)
                    )

                    # D1 and D2 matrices
                    sub_matrix_D1 = hstack(
                        [
                            eye(n_x * n_y * (n_z - 1)),
                            csr_matrix((n_x * n_y * (n_z - 1), n_x * n_y)),
                        ]
                    )
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack(
                        [
                            csr_matrix((n_x * n_y * (n_z - 1), n_x * n_y)),
                            eye(n_x * n_y * (n_z - 1)),
                        ]
                    )
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute beta
                    dx = submesh.d_edges_z
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_x * n_y, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta))

                    # Padding matrix
                    sub_matrix = vstack(
                        [
                            csr_matrix((n_x * n_y, n_x * n_y * (n_z - 1))),
                            eye(n_x * n_y * (n_z - 1)),
                            csr_matrix((n_x * n_y, n_x * n_y * (n_z - 1))),
                        ]
                    )
                    matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

                    return (
                        pybamm.Matrix(edges_matrix) @ array
                        + pybamm.Matrix(matrix) @ D_eff
                    )

            elif shift_key == "edge to node":
                if direction == "x":
                    sub_matrix_D1 = hstack([eye(n_x), csr_matrix((n_x, 1))])
                    sub_matrix_D1 = block_diag((sub_matrix_D1,) * (n_y * n_z))
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack([csr_matrix((n_x, 1)), eye(n_x)])
                    sub_matrix_D2 = block_diag((sub_matrix_D2,) * (n_y * n_z))
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute weight beta
                    dx0 = submesh.nodes_x[0] - submesh.edges_x[0]
                    dxN = submesh.edges_x[-1] - submesh.nodes_x[-1]
                    dx = np.concatenate(([dx0], submesh.d_nodes_x, [dxN]))
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_y * n_z, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    # Compute harmonic mean on nodes
                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta) + 1e-16)
                    return D_eff

                elif direction == "y":
                    # Implement harmonic mean edge to node for y-direction
                    sub_matrix_D1 = hstack(
                        [eye(n_x * n_y), csr_matrix((n_x * n_y, n_x))]
                    )
                    sub_matrix_D1 = block_diag((sub_matrix_D1,) * n_z)
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack(
                        [csr_matrix((n_x * n_y, n_x)), eye(n_x * n_y)]
                    )
                    sub_matrix_D2 = block_diag((sub_matrix_D2,) * n_z)
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute beta for y-direction
                    dx0 = submesh.nodes_y[0] - submesh.edges_y[0]
                    dxN = submesh.edges_y[-1] - submesh.nodes_y[-1]
                    dx = np.concatenate(([dx0], submesh.d_nodes_y, [dxN]))
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_x, axis=0)
                    sub_beta = np.tile(sub_beta, (n_z, 1))
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta) + 1e-16)
                    return D_eff

                elif direction == "z":
                    # Implement harmonic mean edge to node for z-direction
                    sub_matrix_D1 = hstack(
                        [eye(n_x * n_y * n_z), csr_matrix((n_x * n_y * n_z, n_x * n_y))]
                    )
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    sub_matrix_D2 = hstack(
                        [csr_matrix((n_x * n_y * n_z, n_x * n_y)), eye(n_x * n_y * n_z)]
                    )
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute beta for z-direction
                    dx0 = submesh.nodes_z[0] - submesh.edges_z[0]
                    dxN = submesh.edges_z[-1] - submesh.nodes_z[-1]
                    dx = np.concatenate(([dx0], submesh.d_nodes_z, [dxN]))
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_x * n_y, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta) + 1e-16)
                    return D_eff
            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")

        # Validate inputs
        if shift_key not in ["node to edge", "edge to node"]:
            raise ValueError(f"shift key '{shift_key}' not recognised")
        if method not in ["arithmetic", "harmonic"]:
            raise ValueError(f"method '{method}' not recognised")
        if direction not in ["x", "y", "z"]:
            raise ValueError(f"direction '{direction}' not recognised")

        # If discretised_symbol evaluates to number there is no need to average
        if discretised_symbol.size == 1:
            out = discretised_symbol
        elif method == "arithmetic":
            out = arithmetic_mean(discretised_symbol, direction)
        elif method == "harmonic":
            out = harmonic_mean(discretised_symbol, direction)
        else:
            raise ValueError(f"method '{method}' not recognised")

        return out

    def upwind_or_downwind(
        self,
        symbol,
        discretised_symbol,
        bcs,
        x_direction,
        y_direction,
        z_direction=None,
    ):
        """
        3D upwind/downwind: for each axis, either insert a Dirichlet ghost layer
        on the specified face (if direction is "upwind"/"downwind") or else average
        from nodes to edges (if direction is None).

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised.
        discretised_symbol : :class:`pybamm.Vector`
            The already discretised symbol (no ghost cells).
        bcs : dict
            A dictionary mapping `symbol` → {face_name: (value, "Dirichlet"/"Neumann"), …}.
            Valid face names: "left", "right", "front", "back", "bottom", "top".
        x_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a left‐face Dirichlet ghost; if "downwind", insert a right‐face.
            If None, perform node→edge averaging in x.
        y_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a front‐face Dirichlet ghost; if "downwind", insert a back‐face.
            If None, perform node→edge averaging in y.
        z_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a bottom‐face Dirichlet ghost; if "downwind", insert a top‐face.
            If None, perform node→edge averaging in z.

        Returns
        -------
        :class:`pybamm.VectorField3D`
            A 3D vector field whose x, y, z components have been upwinded (i.e. ghost‐cell
            Dirichlet if requested, or else averaged from nodes to edges).
        """
        if symbol not in bcs:
            raise pybamm.ModelError(f"No boundary conditions defined for {symbol!r}.")

        # Determine which face (if any) to use in each axis
        if x_direction == "upwind":
            x_face = "left"
        elif x_direction == "downwind":
            x_face = "right"
        elif x_direction is None:
            x_face = None
        else:
            raise ValueError(
                f"x_direction must be 'upwind', 'downwind', or None, not '{x_direction}'"
            )

        if y_direction == "upwind":
            y_face = "front"
        elif y_direction == "downwind":
            y_face = "back"
        elif y_direction is None:
            y_face = None
        else:
            raise ValueError(
                f"y_direction must be 'upwind', 'downwind', or None, not '{y_direction}'"
            )

        if z_direction == "upwind":
            z_face = "bottom"
        elif z_direction == "downwind":
            z_face = "top"
        elif z_direction is None:
            z_face = None
        else:
            raise ValueError(
                f"z_direction must be 'upwind', 'downwind', or None, not '{z_direction}'"
            )

        # X-component
        if x_face is not None:
            if x_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{x_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][x_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{x_face}' for x upwind/downwind."
                )
            bc_subset = {x_face: (val, bc_type)}
            out_x, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_x = self.node_to_edge(discretised_symbol, direction="x")

        # Y-component
        if y_face is not None:
            if y_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{y_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][y_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{y_face}' for y upwind/downwind."
                )
            bc_subset = {y_face: (val, bc_type)}
            out_y, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_y = self.node_to_edge(discretised_symbol, direction="y")

        # Z-component
        if z_face is not None:
            if z_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{z_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][z_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{z_face}' for z upwind/downwind."
                )
            bc_subset = {z_face: (val, bc_type)}
            out_z, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_z = self.node_to_edge(discretised_symbol, direction="z")

        return pybamm.VectorField3D(out_x, out_y, out_z)
